#!/usr/bin/env bash
# To run:

# cd ~/Dropbox/repos/visdial/; bash train_lf_baselines.sh

# Overfit Discriminative

#cd ~/datasets/myvisdial/checkpoints/tmp
#tensorboard --logdir . --port 8008
#fw 8008 8008

CUDA_VISIBLE_DEVICES='1' python train.py \
--validate \
--overfit \
--gpu-ids 1 \
--cpu-workers 1 \
--comet-name 'test' \
--config-yml 'configs/lf_disc_faster_rcnn_x101.yml' \
--val-json '/home/ubuntu/datasets/myvisdial/data/visdial_1.0_val.json' \
--train-json '/home/ubuntu/datasets/myvisdial/data/visdial_1.0_train.json' \
--val-dense-json '/home/ubuntu/datasets/myvisdial/data/visdial_1.0_val_dense_annotations.json' \
--monitor-path '/home/ubuntu/datasets/myvisdial/checkpoints/tmp/lf_disc_monitor.pkl' \
--save-dirpath '/home/ubuntu/datasets/myvisdial/checkpoints/tmp' \
--load-pthpath ''


# Train Discriminative

#cd ~/datasets/myvisdial/checkpoints/lf_disc/reproduced; tensorboard --logdir . --port 8008
#fw 8008 8008

#CUDA_VISIBLE_DEVICES='0' python train.py \
#--validate \
#--gpu-ids 0 \
#--cpu-workers 4 \
# --comet-name 'visdial-baselines' \
#--config-yml 'configs/lf_disc_faster_rcnn_x101.yml' \
#--save-dirpath '/home/ubuntu/datasets/myvisdial/checkpoints/lf_disc/reproduced' \
#--train-json '/home/ubuntu/datasets/myvisdial/data/visdial_1.0_train.json' \
#--val-json '/home/ubuntu/datasets/myvisdial/data/visdial_1.0_val.json' \
#--val-dense-json '/home/ubuntu/datasets/myvisdial/data/visdial_1.0_val_dense_annotations.json' \
#--monitor-path 'data/lf_disc_monitor.pkl' \
#--load-pthpath '/home/ubuntu/datasets/myvisdial/checkpoints/lf_disc/reproduced/checkpoint_1.pth'
#

# Train Generative

# cd ~/datasets/myvisdial/checkpoints/lf_gen/reproduced
# tensorboard --logdir . --port 8008
# fw 8008 8008

#python train_2.py --overfit --validate --gpu-ids 0 --cpu-workers 1 \
#--config-yml configs/lf_gen_faster_rcnn_x101.yml \
#--save-dirpath /home/ubuntu/datasets/myvisdial/checkpoints/gen_disc \
#--train-json /home/ubuntu/datasets/myvisdial/data/visdial_1.0_train.json \
#--val-json /home/ubuntu/datasets/myvisdial/data/visdial_1.0_val.json \
#--val-dense-json /home/ubuntu/datasets/myvisdial/data/visdial_1.0_val_dense_annotations.json
