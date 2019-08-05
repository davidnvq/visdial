#!/usr/bin/env bash

#!/usr/bin/env sh
#$-pe gpu 8
#$-l gpu=8
#$-j y
#$-cwd
#$-V
#$-o /home/quang/log/v08.log
#$-q main.q@yagi08.vision.is.tohoku

export CUDA_VISIBLE_DEVICES=$SGE_GPU

python net_linear.py \
--config attn_disc_lstm \
>> /home/quang/log/v08.txt
