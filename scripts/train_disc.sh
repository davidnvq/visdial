#!/usr/bin/env bash
# To run:
# cp train_disc.sh train_disc_`date +%d%b%Y`.sh

# cd ~/Dropbox/repos/visdial/; bash train_disc.sh

# OVERFIT Discriminative 1-LSTM

#cd ~/datasets/visdial/checkpoints/tmp
#tensorboard --logdir . --port 8008
#fw 8008 8008

ROOT=/home/ubuntu
DATASET=$ROOT/datasets/visdial
CKPOINT=$ROOT/checkpoints/tmp

CUDA_VISIBLE_DEVICES='0' python ../train.py \
--validate \
--overfit \
--lr 1e-3 \
--gpu-ids 0 \
--cpu-workers 1 \
--comet-name 'test' \
--config-yml 'configs/attn_disc_faster_rcnn_x101.yml' \
--val-json '~/datasets/visdial/data/visdial_1.0_val.json' \
--train-json '~/datasets/visdial/data/visdial_1.0_train.json' \
--val-dense-json '~/datasets/visdial/data/visdial_1.0_val_dense_annotations.json' \
--monitor-path '~/datasets/visdial/checkpoints/tmp/test_disc_monitor.pkl' \
--save-dirpath '~/datasets/visdial/checkpoints/tmp' \
--load-pthpath ''
--image-features-train-h5 '~/datasets/visdial/data/features_faster_rcnn_x101_train.h5' \
--image-features-val-h5 '~/datasets/visdial/data/features_faster_rcnn_x101_val.h5' \
--image-features-test-h5 '~/datasets/visdial/data/features_faster_rcnn_x101_test.h5' \
--word-counts-json '~/datasets/visdial/data/visdial_1.0_word_counts_train.json'

# Train Discriminative
# cd ~/Dropbox/repos/visdial/; bash train_lf_baselines.sh
#cd ~/datasets/visdial/checkpoints/lf_disc/reproduced; tensorboard --logdir . --port 8008
#fw 8008 8008

#CUDA_VISIBLE_DEVICES='0' python train.py \
#--validate \
#--lr 1e-3 \
#--gpu-ids 0 \
#--cpu-workers 4 \
#--comet-name 'visdial-attns' \
#--config-yml 'configs/attn_disc_faster_rcnn_x101.yml' \
#--val-json '~/datasets/visdial/data/visdial_1.0_val.json' \
#--train-json '~/datasets/visdial/data/visdial_1.0_train.json' \
#--val-dense-json '~/datasets/visdial/data/visdial_1.0_val_dense_annotations.json' \
#--monitor-path '~/datasets/visdial/checkpoints/attn_disc/attn_disc_monitor.pkl' \
#--save-dirpath '~/datasets/visdial/checkpoints/attn_disc' \
#--load-pthpath ''
#--image-features-train-h5 '~/datasets/visdial/data/features_faster_rcnn_x101_train.h5' \
#--image-features-val-h5 '~/datasets/visdial/data/features_faster_rcnn_x101_val.h5' \
#--image-features-test-h5 '~/datasets/visdial/data/features_faster_rcnn_x101_test.h5' \
#--word-counts-json '~/datasets/visdial/data/visdial_1.0_word_counts_train.json'
