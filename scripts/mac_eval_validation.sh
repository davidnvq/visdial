# cd ~/Dropbox/repos/visdial/; bash evaluate_lf_baselines.sh

SPLIT=val
ROOT=/Users/quanguet
DATASET=$ROOT/datasets/visdial
CKPOINT=$ROOT/checkpoints/lf_disc
PATH_PROJ=$ROOT/Dropbox/repos/visdial

echo -e "\nSTART EVALUATING....\n"

BATCH_SIZE=2
CONFIG=$PATH_PROJ/configs/lf_disc_faster_rcnn_x101.yml
IMG_FEATURES=$DATASET/features_faster_rcnn_x101_val.h5
JSON_DIALOGS=$DATASET/visdial_1.0_val.json
JSON_DENSE=$DATASET/visdial_1.0_val_dense_annotations.json
JSON_WORD_COUNTS=$DATASET/visdial_1.0_word_counts_train.json
PATH_LOAD=$CKPOINT/lf_disc_faster_rcnn_x101_trainval.pth
PATH_SAVE=$CKPOINT/$SPLIT_ranks.json

CUDA_VISIBLE_DEVICES='0' python $PATH_PROJ/evaluate.py \
--overfit \
--device cpu \
--split $SPLIT \
--config-yml $CONFIG \
--batch-size $BATCH_SIZE \
--json-dense $JSON_DENSE \
--json-dialogs $JSON_DIALOGS \
--image-features-h5 $IMG_FEATURES \
--json-word-counts $JSON_WORD_COUNTS \
--load-pthpath $PATH_LOAD \
--save-ranks-path $PATH_SAVE