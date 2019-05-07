#!/usr/bin/env bash

## Train Discriminative
#python train.py --overfit --validate --gpu-ids 0 --cpu-workers 1 \
#--config-yml configs/lf_disc_faster_rcnn_x101.yml \
#--save-dirpath /home/ubuntu/datasets/myvisdial/checkpoints/lf_disc \
#--train-json /home/ubuntu/datasets/myvisdial/data/visdial_1.0_train.json \
#--val-json /home/ubuntu/datasets/myvisdial/data/visdial_1.0_val.json \
#--val-dense-json /home/ubuntu/datasets/myvisdial/data/visdial_1.0_val_dense_annotations.json

## Train Generative
python train.py --overfit --validate --gpu-ids 0 --cpu-workers 1 \
--config-yml configs/lf_gen_faster_rcnn_x101.yml \
--save-dirpath /home/ubuntu/datasets/myvisdial/checkpoints/gen_disc \
--train-json /home/ubuntu/datasets/myvisdial/data/visdial_1.0_train.json \
--val-json /home/ubuntu/datasets/myvisdial/data/visdial_1.0_val.json \
--val-dense-json /home/ubuntu/datasets/myvisdial/data/visdial_1.0_val_dense_annotations.json

