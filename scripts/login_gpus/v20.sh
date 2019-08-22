#!/usr/bin/env bash

#!/usr/bin/env sh
#$-pe gpu 4
#$-l gpu=4
#$-j y
#$-cwd
#$-V
#$-o /home/quang/workspace/log/temp/v20.log
#$-q main.q@yagi20.vision.is.tohoku

export CUDA_VISIBLE_DEVICES=$SGE_GPU

python net_linear.py >> /home/quang/workspace/log/temp/v20.txt
