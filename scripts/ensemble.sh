#!/usr/bin/env bash

SPLIT='val'
ROOT='/content'
PATH_PROJ=$ROOT/visdial
DATA=$ROOT/datasets/visdial
CKPT=/content/gdrive/My\ Drive/checkpoints/lf_disc/may13
CONFIG=$PATH_PROJ/configs/lf_disc_faster_rcnn_x101.yml

python $PATH_PROJ/ensemble.py \
--overfit \
--split val \
--config-ymls $CONFIG $CONFIG \
--json-dialogs $DATA/visdial_1.0_val.json \
--json-dense $DATA/visdial_1.0_val_dense_annotations.json \
--load-pthpaths $CKPT/checkpoint_best_mean.pth $CKPT/checkpoint_best_ndcg.pth \
--image-features-h5 $DATA/features_faster_rcnn_x101_val.h5 \
--save-ranks-path $CKPT/val_rank.json \
--json-word-counts $DATA/visdial_1.0_word_counts_train.json


