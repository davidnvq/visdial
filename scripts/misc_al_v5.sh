#!/usr/bin/env sh
#$-pe gpu 4
#$-l gpu=4
#$-j y
#$-cwd
#$-V
#$-o /home/quang/log/log_attn_misc_lstm_v5_yagi16.log
#$-q main.q@yagi16.vision.is.tohoku

export CUDA_VISIBLE_DEVICES=$SGE_GPU

python /home/quang/repos/visdial_v5/train.py \
--config attn_misc_lstm \
>> /home/quang/log/output_attn_misc_lstm_v5_yagi16.txt