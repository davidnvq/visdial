#!/usr/bin/env bash

#python evaluate.py \
#--split test \
#--config attn_misc_lstm \
#--decoder_type disc \
#--weights /home/quanguet/checkpoints/visdial/attn_misc_lstm_v2_24_July_seed_1994/fixed_checkpoint_21.pth \
#--save-ranks-path /home/quanguet/checkpoints/visdial/attn_misc_lstm_v2_24_July_seed_1994/test/test_{}_ranks.json
#
#python evaluate.py \
#--split test \
#--config attn_misc_lstm \
#--decoder_type gen \
#--weights /home/quanguet/checkpoints/visdial/attn_misc_lstm_v2_24_July_seed_1994/fixed_checkpoint_21.pth \
#--save-ranks-path /home/quanguet/checkpoints/visdial/attn_misc_lstm_v2_24_July_seed_1994/test/test_{}_ranks.json

python evaluate.py \
--split test \
--config attn_misc_lstm \
--decoder_type disc \
--weights /home/quanguet/checkpoints/visdial/attn_misc_lstm_v2_24_July_seed_1994/finetune/checkpoint_5.pth \
--save-ranks-path /home/quanguet/checkpoints/visdial/attn_misc_lstm_v2_24_July_seed_1994/finetune/test/test_ckpt5_{}_ranks.json

python evaluate.py \
--split test \
--config attn_misc_lstm \
--decoder_type disc \
--weights /home/quanguet/checkpoints/visdial/attn_misc_lstm_v2_24_July_seed_1994/finetune/checkpoint_6.pth \
--save-ranks-path /home/quanguet/checkpoints/visdial/attn_misc_lstm_v2_24_July_seed_1994/finetune/test/test_ckpt6_{}_ranks.json

#python evaluate.py \
#--split test \
#--config attn_misc_lstm \
#--decoder_type misc \
#--weights /home/quanguet/checkpoints/visdial/attn_misc_lstm_v2_24_July_seed_1994/finetune/fixed_checkpoint_21.pth \
#--save-ranks-path /home/quanguet/checkpoints/visdial/attn_misc_lstm_v2_24_July_seed_1994/test/test_{}_ranks.json