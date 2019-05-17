#!/usr/bin/env bash
# To run:
# bash /content/visdial/scripts/colab_train_lf_gen.sh
ROOT='/Users/quanguet'
PATH_PROJ=$ROOT/Dropbox/repos/visdial

python $PATH_PROJ/scripts/colab_train.py \
--gdrive \
--overfit \
--num-epochs 10 \
--lr 1e-3 \
--lr-steps 5 \
--encoder 'lf' \
--decoder 'gen' \
--ckpt-path 'may14' \
--load-pthpath 'checkpoint_old_9.pth'