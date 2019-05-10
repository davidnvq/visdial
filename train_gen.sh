#!/usr/bin/env bash
# To run:

# cd ~/Dropbox/repos/visdial/; bash train_gen.sh


# Train Generative

# cd ~/datasets/visdial/checkpoints/lf_gen/reproduced
# tensorboard --logdir . --port 8008
# fw 8008 8008

#CUDA_VISIBLE_DEVICES='1' python train.py \
#--validate \
#--overfit \
#--gpu-ids 1 \
#--cpu-workers 4 \
#--comet-name 'test' \
#--config-yml 'configs/attn_gen_faster_rcnn_x101.yml' \
#--val-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_val.json' \
#--train-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_train.json' \
#--val-dense-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_val_dense_annotations.json' \
#--save-dirpath '/home/ubuntu/datasets/visdial/checkpoints/tmp' \
#--load-pthpath ''

CUDA_VISIBLE_DEVICES='1' python train.py \
--validate \
--lr 1e-3 \
--gpu-ids 1 \
--cpu-workers 4 \
--comet-name 'visdial-attns' \
--config-yml 'configs/attn_gen_faster_rcnn_x101.yml' \
--val-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_val.json' \
--train-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_train.json' \
--val-dense-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_val_dense_annotations.json' \
--save-dirpath '/home/ubuntu/datasets/visdial/checkpoints/attn_gen' \
--load-pthpath ''