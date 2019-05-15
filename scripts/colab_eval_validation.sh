#!/usr/bin/env bash
# bash /content/visdial/scripts/colab_eval_validation.sh

SPLIT='val'
ENCODER='lf'
DECODER='disc'
python colab_eval.py \
--overfit \
--gdrive \
--split $SPLIT \
--encoder $ENCODER \
--decoder $DECODER \
--ckpt-path 'may13' \
--load-pthpath 'checkpoint_9.pth'