#!/usr/bin/env sh
#$-pe gpu 4
#$-l gpu=4
#$-j y
#$-cwd
#$-V
#$-o /home/quang/workspace/log/qsub/train_simple/s03.log
#$-q main.q@yagi17.vision.is.tohoku

export CUDA_VISIBLE_DEVICES=$SGE_GPU
python /home/quang/workspace/repos/visdial/train.py \
--config_name train_simple/12epochs/s03_simple_12epochs \
--decoder_type misc \
--init_lr 0.001 \
--scheduler_type "LinearLR" \
--batch_size 8 \
--num_epochs 12 \
--num_samples 123287 \
--ls_epsilon 0.0 \
--dropout 0.1 \
--ca_has_layer_norm \
--ca_has_shared_attns \
--ca_num_cross_attns 1 \
--txt_has_layer_norm \
--txt_has_pos_embedding \
--val_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_36_36.h5" \
--train_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/trainval_resnet101_faster_rcnn_genome__num_boxes_36_36.h5" \
>> /home/quang/workspace/log/qsub/train_simple/s03.txt