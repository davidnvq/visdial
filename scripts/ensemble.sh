#!/usr/bin/env bash

SPLIT='val'
ROOT='/content'
PATH_PROJ=$ROOT/visdial
DATA=$ROOT/datasets/visdial
CONFIG=$PATH_PROJ/configs/lf_disc_faster_rcnn_x101.yml

python $PATH_PROJ/ensemble.py \
--split $SPLIT \
--json-dialogs $DATA/visdial_1.0_val.json \
--json-dense $DATA/visdial_1.0_val_dense_annotations.json \
--image-features-h5 $DATA/features_faster_rcnn_x101_val.h5 \
--json-word-counts $DATA/visdial_1.0_word_counts_train.json


##!/usr/bin/env bash
#SPLIT='test'
#ROOT='/content'
#PATH_PROJ=$ROOT/visdial
#DATA=$ROOT/datasets/visdial
#CONFIG=$PATH_PROJ/configs/lf_disc_faster_rcnn_x101.yml
#
#python $PATH_PROJ/ensemble.py \
#--split $SPLIT \
#--json-dialogs $DATA/visdial_1.0_test.json \
#--image-features-h5 $DATA/features_faster_rcnn_x101_test.h5 \
#--json-word-counts $DATA/visdial_1.0_word_counts_train.json
#



