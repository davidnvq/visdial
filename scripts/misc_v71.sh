#!/usr/bin/env sh
#$-pe gpu 4
#$-l gpu=4
#$-j y
#$-cwd
#$-V
#$-o /home/quang/log/v71log.txt
#$-q main.q@yagi11.vision.is.tohoku

export CUDA_VISIBLE_DEVICES=$SGE_GPU
python /home/quang/repos/visdial/train.py \
--config attn_misc_lstm \
>> /home/quang/log/v71out.txt