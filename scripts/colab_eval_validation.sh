#!/usr/bin/env bash

SPLIT='val'
ENCODER='lf'
DECODER='attn'
wcolab_eval.py \
--overfit \
--gdrive \
--split $SPLIT \
--encoder $ENCODER \
--decoder $DECODER \
--ckpt-path 'may13' \
--load-pthpath 'checkpoint_9.pth'