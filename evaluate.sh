#!/usr/bin/env bash

python evaluate.py \
--split val \
--config attn_misc_lstm \
--decoder_type misc \
--weights /home/quanguet/checkpoints/visdial/attn_misc_lstm_v2_24_July_seed_1994/fixed_checkpoint_21.pth \
--save-ranks-path /home/quanguet/checkpoints/visdial/attn_misc_lstm_v2_24_July_seed_1994/gen_val_ranks.json