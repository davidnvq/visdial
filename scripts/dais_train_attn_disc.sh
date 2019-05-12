#!/usr/bin/env bash
# To run:
# bash /home/ubuntu/Dropbox/repos/visdial/scripts/dais_train_attn_disc.sh

OVERFIT=true
ROOT=/home/ubuntu
DATASET=$ROOT/datasets/visdial
CKPOINT=$ROOT/checkpoints/attn_disc

PATH_LOAD=$CKPOINT/attn_disc_faster_rcnn_x101_trainval.pth
PATH_PROJ=$ROOT/Dropbox/repos/visdial
CONFIG=$PATH_PROJ/configs/attn_disc_faster_rcnn_x101.yml

if [ $OVERFIT = true ]; then
    PATH_SAVE=$CKPOINT/tmp
    COMET=test
else
    PATH_SAVE=$CKPOINT/attn_disc_`date +%d%b%Y`
    COMET=visdial_attn_disc
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
cp $PATH_PROJ/scripts/dais_train_attn_gen.sh $PATH_SAVE/


# Overfit
python $PATH_PROJ/train.py \
--validate \
--overfit \
--gpu-ids 0 1 \
--cpu-workers 4 \
--batch-size 48 \
--num-epochs 10 \
--lr 1e-3 \
--step-size 2 \
--comet-name $COMET \
--config-yml $CONFIG \
--image-features-tr-h5 ${FILE[1]} \
--image-features-va-h5 ${FILE[2]} \
--json-word-counts ${FILE[3]} \
--json-train ${FILE[4]} \
--json-val ${FILE[5]} \
--json-val-dense ${FILE[6]} \
--save-dirpath $PATH_SAVE \

echo $PATH_SAVE