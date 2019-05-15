#!/usr/bin/env bash

SPLIT='test'
ENCODER='lf'
DECODER='attn'
ROOT='/Users/quanguet'
PATH_PROJ=$ROOT/Dropbox/repos/visdial

python $PATH_PROJ/scripts/colab_eval.py \
--overfit \
--gdrive \
--split $SPLIT \
--encoder $ENCODER \
--decoder $DECODER \
--ckpt-path 'may13' \
--load-pthpath 'checkpoint_9.pth'