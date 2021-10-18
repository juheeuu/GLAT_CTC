# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import math
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.nat.nonautoregressive_transformer import NATransformerEncoder, NATransformerModel
from .nat_modules import NATransformerDecoder, get_aligned_target
import logging
import random
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def torch_seed(seed):
    state = torch.random.get_rng_state()
    state_cuda = torch.cuda.random.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        torch.cuda.random.set_rng_state(state_cuda)
        
def initialize_output_tokens(self, encoder_out, src_tokens):

    batch_size = src_tokens.size(0)
    src_lengths = src_tokens.ne(self.pad).sum(dim=-1)
    
    src_lengths_up = src_lengths * self.src_upsampling_rate
    max_length = src_lengths_up.max()
    idx_length = utils.new_arange(src_tokens, max_length)

    initial_output_tokens = src_tokens.new_zeros(
        src_tokens.size(0), max_length
    ).fill_(self.pad)
    initial_output_tokens.masked_fill_(
        idx_length[None, :] < src_lengths_up[:, None], self.unk
    )
    initial_output_tokens.scatter_(1, src_lengths_up[:, None] - 1, self.eos)

    initial_output_scores = initial_output_tokens.new_zeros(
        *initial_output_tokens.size()
    ).type_as(encoder_out["encoder_out"][0])

    return DecoderOut(
        output_tokens=initial_output_tokens,
        output_scores=initial_output_scores,
        attn=None,
        step=0,
        max_step=0,
        history=None,
    )


@register_model("glat_ctc")
class GlatCTC(FairseqNATModel):
    forward_decoder = NATransformerModel.forward_decoder
    initialize_output_tokens = initialize_output_tokens
    regenerate_length_beam = NATransformerModel.regenerate_length_beam

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        
        self.src_upsampling_rate = getattr(args, "src_upsampling_rate", 3)

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--src-upsampling-rate",
            type=int,
            help="source upsampling",
        )
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--sg-length-pred",
            action="store_true",
            help="stop the gradients back-propagated from the length predictor",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )

    @classmethod
    def build_encoder(cls, args, tgt_dict, embed_tokens):
        encoder = NATransformerEncoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat=None, **kwargs
    ):        
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        
        # Upsampling! 
        src_lengths_up = src_lengths * self.src_upsampling_rate
        max_length = src_lengths_up.max()
        idx_length = utils.new_arange(src_tokens, max_length)

        prev_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        prev_output_tokens.masked_fill_(
            idx_length[None, :] < src_lengths_up[:, None], self.unk
        )
        prev_output_tokens.scatter_(1, src_lengths_up[:, None] - 1, self.eos)
        
        rand_seed = random.randint(0, 19260817)
        # glancing sampling
        glat_info = None
        if glat and tgt_tokens is not None:
            with torch.no_grad():
                with torch_seed(rand_seed):
                    word_ins_out = self.decoder(
                        normalize=False,
                        prev_output_tokens=prev_output_tokens,
                        encoder_out=encoder_out,
                    )
                
                # calculate viterbi alignment ..
                # word_ins_out : [b, t_s * upsampling_rate, c]
                # tgt_tokens : [b, t_t] 
                log_probs = torch.log_softmax(word_ins_out, dim=-1).transpose(0, 1)
                target_lengths = tgt_tokens.ne(self.pad).sum(dim=-1) 
                
                vb_tgt_tokens = get_aligned_target(log_probs, tgt_tokens, src_lengths_up, target_lengths)
                nonpad_positions = vb_tgt_tokens.ne(self.pad)
                seq_lens = (nonpad_positions).sum(1)
                    
                pred_tokens = word_ins_out.argmax(-1)
                
                same_num = ((pred_tokens == vb_tgt_tokens) & nonpad_positions).sum(1)
                
                input_mask = torch.ones_like(nonpad_positions)
                bsz, seq_len = tgt_tokens.size()
                for li in range(bsz):
                    target_num = (((seq_lens[li] - same_num[li].sum()).float()) * glat['context_p']).long()
                    if target_num > 0:
                        input_mask[li].scatter_(dim=0, index=torch.randperm(seq_lens[li])[:target_num].cuda(), value=0)
                        
                input_mask = input_mask.eq(1)
                input_mask = input_mask.masked_fill(~nonpad_positions,False)
                glat_prev_output_tokens = prev_output_tokens.masked_fill(~input_mask, 0) + vb_tgt_tokens.masked_fill(input_mask, 0)
                prev_output_tokens = glat_prev_output_tokens
                
                glat_info = {
                    "glat_accu": (same_num.sum() / seq_lens.sum()).item(),
                    "glat_context_p": glat['context_p'],
                }
                
        with torch_seed(rand_seed):
            word_ins_out = self.decoder(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
            )

        ret = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
        }
        if glat_info is not None:
            ret.update(glat_info)
        return ret
    
    

@register_model_architecture(
    "glat_ctc", "glat_ctc_6e6d512"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture(
    "glat_ctc", "glat_ctc"
)
def glat_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", args.encoder_embed_dim*4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", args.encoder_embed_dim//64)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.decoder_embed_dim*4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", args.decoder_embed_dim//64)
    base_architecture(args)

@register_model_architecture(
    "glat_ctc", "glat_ctc_base"
)
def base_architecture2(args):
    base_architecture(args)
