import torch
import math
import torch.nn.functional as F
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATEncoder, FairseqNATModel, ensemble_decoder, ensemble_encoder
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor

@torch.jit.script
def get_aligned_target(log_probs : torch.Tensor, targets : torch.Tensor, input_lengths : torch.Tensor, 
                       target_lengths : torch.Tensor, blank: int = 0, finfo_min_fp32: float = torch.finfo(torch.float32).min, 
                       finfo_min_fp16: float = torch.finfo(torch.float16).min):

    input_time_size, batch_size = log_probs.shape[:2]
    B = torch.arange(batch_size, device = input_lengths.device)

    _t_a_r_g_e_t_s_ = torch.cat([
        torch.stack([torch.full_like(targets, blank), targets], dim = -1).flatten(start_dim = -2),
        torch.full_like(targets[:, :1], blank)
    ], dim = -1)
    diff_labels = torch.cat([
        torch.as_tensor([[False, False]], device = targets.device).expand(batch_size, -1),
        _t_a_r_g_e_t_s_[:, 2:] != _t_a_r_g_e_t_s_[:, :-2]
    ], dim = 1)


    zero_padding, zero = 2, torch.tensor(finfo_min_fp16 if log_probs.dtype == torch.float16 else finfo_min_fp32, device = log_probs.device, dtype = log_probs.dtype)
    padded_t = zero_padding + _t_a_r_g_e_t_s_.shape[-1]
    log_alpha = torch.full((batch_size, padded_t), zero, device = log_probs.device, dtype = log_probs.dtype)
    log_alpha[:, zero_padding + 0] = log_probs[0, :, blank]
    log_alpha[:, zero_padding + 1] = log_probs[0, B, _t_a_r_g_e_t_s_[:, 1]]

    packnibbles = 1
    backpointers_shape = [len(log_probs), batch_size, int(math.ceil(padded_t / packnibbles))]
    backpointers = torch.zeros(backpointers_shape, device = log_probs.device, dtype = torch.uint8)
    backpointer = torch.zeros(backpointers_shape[1:], device = log_probs.device, dtype = torch.uint8)

    for t in range(1, input_time_size):
        prev = torch.stack([log_alpha[:, 2:], log_alpha[:, 1:-1], torch.where(diff_labels, log_alpha[:, :-2], zero)])
        log_alpha[:, zero_padding:] = log_probs[t].gather(-1, _t_a_r_g_e_t_s_) + prev.logsumexp(dim = 0)
        backpointer[:, zero_padding:(zero_padding + prev.shape[-1] )] = prev.argmax(dim = 0)
        backpointers[t] = backpointer

    l1l2 = log_alpha.gather(-1, torch.stack([zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], dim = -1))

    path = torch.zeros(input_time_size, batch_size, device = log_alpha.device, dtype = torch.long)
    path[input_lengths - 1, B] = zero_padding + target_lengths * 2 - 1 + l1l2.argmax(dim = -1)

    for t in range(input_time_size - 1, 0, -1):
        indices = path[t]
        backpointer = backpointers[t]
        path[t - 1] += indices - backpointer.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    path = path - zero_padding 
    path = path.transpose(0, 1) #[b, t_t]
    path_mask = (torch.arange(input_time_size).unsqueeze(0).repeat(batch_size, 1).to(path.device) < input_lengths.unsqueeze(-1))
    path = path * path_mask.long()
    return _t_a_r_g_e_t_s_.gather(-1, path)


class NATransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.encoder_embed_dim = args.encoder_embed_dim
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
        self.src_embedding_copy = getattr(args, "src_embedding_copy", False)
        self.vae = getattr(args, "vae", False)
        if self.src_embedding_copy:
            self.copy_attn = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, **unused):
        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out

    @ensemble_decoder
    def forward_length(self, normalize, encoder_out):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        enc_feats = _mean_pooling(enc_feats, src_masks)
        if self.sg_length_pred:
            enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )
        # embedding
        if embedding_copy:
            src_embd = encoder_out["encoder_out"][0].transpose(0, 1)
            if len(encoder_out["encoder_padding_mask"]) > 0:
                src_mask = encoder_out["encoder_padding_mask"][0]
            else:
                src_mask = None

            bsz, seq_len = prev_output_tokens.size()
            attn_score = torch.bmm(self.copy_attn(positions),
                                   (src_embd + encoder_out['encoder_pos'][0]).transpose(1, 2))
            if src_mask is not None:
                attn_score = attn_score.masked_fill(src_mask.unsqueeze(1).expand(-1, seq_len, -1), float('-inf'))
            attn_weight = F.softmax(attn_score, dim=-1)
            x = torch.bmm(attn_weight, src_embd)
            mask_target_x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)
            output_mask = prev_output_tokens.eq(self.unk)
            
            cat_x = torch.cat([mask_target_x.unsqueeze(2), x.unsqueeze(2)], dim=2).view(-1, x.size(2)) 
            x = cat_x.index_select(dim=0, index=torch.arange(bsz * seq_len).cuda() * 2 +
                                                output_mask.view(-1).long()).reshape(bsz, seq_len, x.size(2))
            
            
        else:

            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        positions = positions.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            if positions is not None:
                x += positions
            x = self.dropout_module(x)

            x, attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    def forward_embedding(self, prev_output_tokens, states=None):
        # embed tokens
        if states is None:
            x = self.embed_tokens(prev_output_tokens)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
        else:
            x = states

        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask

