#!/usr/bin/env bash
# To run:

# cd ~/Dropbox/repos/visdial/; bash experiment.sh

# Overfit Discriminative

#cd ~/datasets/myvisdial/checkpoints/tmp
#tensorboard --logdir . --port 8008
#fw 8008 8008

CUDA_VISIBLE_DEVICES='1' python train_experiment.py \
--validate \
--gpu-ids 1 \
--cpu-workers 1 \
--config-yml 'configs/lf_gen_faster_rcnn_x101.yml' \
--val-json '/home/ubuntu/datasets/myvisdial/data/visdial_1.0_val.json' \
--train-json '/home/ubuntu/datasets/myvisdial/data/visdial_1.0_train.json' \
--val-dense-json '/home/ubuntu/datasets/myvisdial/data/visdial_1.0_val_dense_annotations.json' \
--save-dirpath '/home/ubuntu/datasets/myvisdial/checkpoints/lf_gen/may8' \
--load-pthpath ''