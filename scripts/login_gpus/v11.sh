#!/usr/bin/env bash

#!/usr/bin/env sh
#$-pe gpu 4
#$-l gpu=4
#$-j y
#$-cwd
#$-V
#$-o /home/quang/log/v11.log
#$-q main.q@yagi11.vision.is.tohoku

export CUDA_VISIBLE_DEVICES=$SGE_GPU

python net_linear.py \
--config attn_disc_lstm \
>> /home/quang/log/v11.txt