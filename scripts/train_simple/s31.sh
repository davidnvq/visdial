#!/usr/bin/env sh
#$-pe gpu 4
#$-l gpu=4
#$-j y
#$-cwd
#$-V
#$-o /home/quang/workspace/log/qsub/train_simple/s31.log
#$-q main.q@yagi17.vision.is.tohoku

export CUDA_VISIBLE_DEVICES=$SGE_GPU
/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/train.py \
--config_name train_simple/s31 \
--decoder_type misc \
--init_lr 0.001 \
--scheduler_type "LinearLR" \
--batch_size 8 \
--num_epochs 15 \
--num_samples 123287 \
--ls_epsilon 0.0 \
--milestone_steps 3 5 7 9 11 13 \
--dropout 0.1 \
--img_has_bboxes \
--ca_has_layer_norm \
--ca_num_cross_attns 2 \
--ca_has_residual \
--ca_every_head_attn \
--txt_has_layer_norm \
--txt_has_decoder_layer_norm \
--txt_has_pos_embedding \
--val_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_100_100.h5" \
--train_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/trainval_resnet101_faster_rcnn_genome__num_boxes_100_100.h5" \
>> /home/quang/workspace/log/qsub/train_simple/s31.txt