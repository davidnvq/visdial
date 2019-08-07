#!/usr/bin/env sh
#$-pe gpu 4
#$-l gpu=4
#$-j y
#$-cwd
#$-V
#$-o /home/quang/log/log_v1.0.txt
#$-q main.q@yagi08.vision.is.tohoku

export CUDA_VISIBLE_DEVICES=$SGE_GPU
python /home/quang/repos/visdial/train.py \
--config attn_misc_lstm \
>> /home/quang/log/out_v1.0.txt