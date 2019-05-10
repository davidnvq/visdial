#!/usr/bin/env bash
# To run:

# cd ~/Dropbox/repos/visdial/; bash train_disc.sh

# OVERFIT Discriminative 1-LSTM

#cd ~/datasets/visdial/checkpoints/tmp
#tensorboard --logdir . --port 8008
#fw 8008 8008

#CUDA_VISIBLE_DEVICES='1' python train.py \
#--validate \
#--overfit \
#--gpu-ids 1 \
#--cpu-workers 1 \
#--comet-name 'test' \
#--config-yml 'configs/attn_disc_faster_rcnn_x101.yml' \
#--val-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_val.json' \
#--train-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_train.json' \
#--val-dense-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_val_dense_annotations.json' \
#--monitor-path '/home/ubuntu/datasets/visdial/checkpoints/tmp/test_disc_monitor.pkl' \
#--save-dirpath '/home/ubuntu/datasets/visdial/checkpoints/tmp' \
#--load-pthpath ''
#

# Train Discriminative
# cd ~/Dropbox/repos/visdial/; bash train_lf_baselines.sh
#cd ~/datasets/visdial/checkpoints/lf_disc/reproduced; tensorboard --logdir . --port 8008
#fw 8008 8008

CUDA_VISIBLE_DEVICES='0' python train.py \
--validate \
--lr 1e-3 \
--gpu-ids 0 \
--cpu-workers 4 \
--comet-name 'visdial-attns' \
--config-yml 'configs/attn_disc_faster_rcnn_x101.yml' \
--val-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_val.json' \
--train-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_train.json' \
--val-dense-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_val_dense_annotations.json' \
--monitor-path '/home/ubuntu/datasets/visdial/checkpoints/attn_disc/attn_disc_monitor.pkl' \
--save-dirpath '/home/ubuntu/datasets/visdial/checkpoints/attn_disc' \
--load-pthpath ''
