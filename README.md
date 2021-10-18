# GLAT-CTC-pytorch 

Unofficial implementation of ACL2021 paper "Glancing Transformer for Non-Autoregressive Neural Machine Translation" in pytorch 

Some part of the codes of in this repository come from 
- https://github.com/FLC777/GLAT (Original GLAT)
- https://github.com/vadimkantorov/ctc (CTC Alignments)

# Usage 

### Dataset 

Build your own distillation data with fairseq 

### Training 

``` bash 
python3 train.py $DATA_DIR --save-dir $SAVE_DIR --arch glat_ctc \
    --criterion glat_ctc_loss --task translation_lev_ctc
    --noise full_mask --share-all-embeddings \
    --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-6  --max-tokens 4096 --weight-decay 0.01 --dropout 0.1 \
    --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 \
    --max-source-positions 1000 --max-target-positions 1000 --max-update 300000 --seed 0 --clip-norm 5 \
     --src-embedding-copy --length-loss-factor 0.05 --log-interval 1000  \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu  \
    --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu --user-dir glat_plugins
```


### Generation 
``` bash 
python3 glat_plugins/generate_ctc.py $DATA_DIR --path $CKPT_DIR --user-dir glat_plugins --remove-bpe \
    --task translation_lev_ctc  --max-sentences 20 --source-lang $SRC_LAN --target-lang $TGT_LAN  \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 --gen-subset test
```


# Experiment Result 

### WMT14 (BLEU Score)

| Model  | WMT14 en-de (distil) | WMT14 de-en (distil)
| ------------- | ------------- | ------------- |
| GLAT  | 24.72  | 29.72  |
| GLAT + CTC  | 25.68  | 30.23  |