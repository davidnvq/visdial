#!/usr/bin/env bash

#!/usr/bin/env sh
#$-pe gpu 8
#$-l gpu=8
#$-j y
#$-cwd
#$-V
#$-o /home/quang/workspace/log/temp/v17.log
#$-q main.q@yagi17.vision.is.tohoku

export CUDA_VISIBLE_DEVICES=$SGE_GPU

python net_linear.py >> /home/quang/workspace/log/temp/v17.txt
