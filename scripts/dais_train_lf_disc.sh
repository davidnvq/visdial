#!/usr/bin/env bash
# To run:
# bash /home/ubuntu/Dropbox/repos/visdial/scripts/dais_train_lf_disc.sh

OVERFIT=false
ROOT=/home/ubuntu
DATASET=$ROOT/datasets/visdial
CKPOINT=$ROOT/checkpoints/lf_disc

PATH_LOAD=$CKPOINT/may13/checkpoint_best_ndcg.pth
PATH_PROJ=$ROOT/Dropbox/repos/visdial
CONFIG=$PATH_PROJ/configs/lf_disc_faster_rcnn_x101.yml

if [ $OVERFIT = true ]; then
    PATH_SAVE=$CKPOINT/tmp
    COMET=test
else
    PATH_SAVE=$CKPOINT/may17
    COMET=visdial-disc
fi

FILE[1]=$DATASET/features_faster_rcnn_x101_train.h5
FILE[2]=$DATASET/features_faster_rcnn_x101_val.h5
FILE[3]=$DATASET/visdial_1.0_word_counts_train.json
FILE[4]=$DATASET/visdial_1.0_train.json
FILE[5]=$DATASET/visdial_1.0_val.json
FILE[6]=$DATASET/visdial_1.0_val_dense_annotations.json

# copy bash file
if [ ! -d $PATH_SAVE ]; then
    mkdir -p $PATH_SAVE
fi
cp $PATH_PROJ/scripts/dais_train_lf_disc.sh $PATH_SAVE/

# Overfit
python $PATH_PROJ/train.py \
--validate \
--resume \
--gpu-ids 0 1 \
--cpu-workers 4 \
--batch-size 30 \
--num-epochs 15 \
--lr 1e-3 \
--step-size 1 \
--comet-name $COMET \
--config-yml $CONFIG \
--image-features-tr-h5 ${FILE[1]} \
--image-features-va-h5 ${FILE[2]} \
--json-word-counts ${FILE[3]} \
--json-train ${FILE[4]} \
--json-val ${FILE[5]} \
--json-val-dense ${FILE[6]} \
--load-pthpath $PATH_LOAD \
--save-dirpath $PATH_SAVE \

echo $PATH_SAVE
ls -l $PATH_SAVE