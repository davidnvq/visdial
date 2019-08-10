#!/usr/bin/env sh
#$-pe gpu 2
#$-l gpu=2
#$-j y
#$-cwd
#$-V
#$-o /home/quang/log/log_v1.2.0.txt
#$-q main.q@yagi03.vision.is.tohoku

export CUDA_VISIBLE_DEVICES=$SGE_GPU
python /home/quang/repos/visdial/train.py \
--config attn_misc_lstm \
>> /home/quang/log/out_v1.2.0.txt