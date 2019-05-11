#!/usr/bin/env bash
# To run:
# cd /content/visdial/scripts; cp train_disc.sh train_disc_`date +%d%b%Y`.sh; bash train_disc_`date +%d%b%Y`.sh
ROOT=/content
DATASET=$ROOT/datasets/visdial
CKPOINT=$ROOT/checkpoints/tmp
PATH_PROJ=$ROOT/visdial

echo -e "\nSTART DOWNLOADING....\n"

bash $PATH_PROJ/scripts/download_data.sh

echo -e "\nSTART TRAINING....\n"

ROOT=/content
DATASET=$ROOT/datasets/visdial
CKPOINT=$ROOT/checkpoints/tmp
PATH_PROJ=$ROOT/visdial

COMET=test
LR=1e-3
BATCH_SIZE=16
NUM_EPOCHS=10
CONFIG=$PATH_PROJ/configs/lf_disc_faster_rcnn_x101.yml


FILE[1]=$DATASET/features_faster_rcnn_x101_train.h5
FILE[2]=$DATASET/features_faster_rcnn_x101_val.h5
FILE[3]=$DATASET/features_faster_rcnn_x101_test.h5
FILE[4]=$DATASET/visdial_1.0_word_counts_train.json
FILE[5]=$DATASET/visdial_1.0_train.json
FILE[6]=$DATASET/visdial_1.0_val.json
FILE[7]=$DATASET/visdial_1.0_val_dense_annotations.json

PATH_LOAD="''"
PATH_SAVE=$CKPOINT/
PATH_MONI=$CKPOINT/monitor.pkl
PATH_PROJ=$ROOT/visdial

# Overfit
python $PATH_PROJ/train.py \
--validate \
--overfit \
--lr $LR \
--comet-name $COMET \
--config-yml $CONFIG \
--image-features-tr-h5 ${FILE[1]} \
--image-features-va-h5 ${FILE[2]} \
--image-features-te-h5 ${FILE[3]} \
--json-word-counts ${FILE[4]} \
--json-train ${FILE[5]} \
--json-val ${FILE[6]} \
--json-val-dense ${FILE[7]} \
--monitor-path $PATH_MONI \
--save-dirpath $PATH_SAVE \
--batch-size $BATCH_SIZE \
--num-epochs $NUM_EPOCHS