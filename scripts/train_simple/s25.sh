#!/usr/bin/env sh
export CUDA_VISIBLE_DEVICES=0,1
python /home/quang/workspace/repos/visdial/train.py \
--seed 2303 \
--config_name train_simple/s25_2303 \
--decoder_type misc \
--init_lr 0.001 \
--scheduler_type "LinearLR" \
--batch_size 16 \
--num_epochs 15 \
--num_samples 123287 \
--milestone_steps 3 5 7 9 11 13 \
--dropout 0.1 \
--img_has_bboxes \
--img_has_classes \
--img_has_attributes \
--ca_has_shared_attns \
--ca_has_layer_norm \
--ca_num_cross_attns 2 \
--ca_has_residual \
--txt_has_layer_norm \
--txt_has_decoder_layer_norm \
--txt_has_pos_embedding \
--val_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_100_100.h5" \
--train_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/trainval_resnet101_faster_rcnn_genome__num_boxes_100_100.h5"

#!/usr/bin/env sh
export CUDA_VISIBLE_DEVICES=0,1
python /home/quang/workspace/repos/visdial/train.py \
--seed 2103 \
--config_name train_simple/s25_2103 \
--decoder_type misc \
--init_lr 0.001 \
--scheduler_type "LinearLR" \
--batch_size 16 \
--num_epochs 15 \
--num_samples 123287 \
--milestone_steps 3 5 7 9 11 13 \
--dropout 0.1 \
--img_has_bboxes \
--img_has_classes \
--img_has_attributes \
--ca_has_shared_attns \
--ca_has_layer_norm \
--ca_num_cross_attns 2 \
--ca_has_residual \
--txt_has_layer_norm \
--txt_has_decoder_layer_norm \
--txt_has_pos_embedding \
--val_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_100_100.h5" \
--train_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/trainval_resnet101_faster_rcnn_genome__num_boxes_100_100.h5"
