#!/usr/bin/env bash
# To run:
# bash /content/visdial/scripts/colab_train_attn_disc.sh
ROOT='/Users/quanguet'
PATH_PROJ=$ROOT/Dropbox/repos/visdial

python $PATH_PROJ/scripts/colab_train.py \
--gdrive \
--overfit \
--num-epochs 15 \
--lr 1e-3 \
--lr-steps 5 \
--encoder 'attn' \
--decoder 'disc' \
--ckpt-path 'may14'