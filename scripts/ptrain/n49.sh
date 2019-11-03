#!/usr/bin/env sh
#$-pe gpu 4
#$-l gpu=4
#$-j y
#$-cwd
#$-V
#$-o /home/quang/workspace/log/qsub/train_simple/n49.log
#$-q main.q@yagi02.vision.is.tohoku

export CUDA_VISIBLE_DEVICES=$SGE_GPU
/home/quang/anaconda3/bin/python ~/workspace/repos/visdial/train.py \
--config_name n49 \
--decoder_type misc \
--init_lr 0.001 \
--scheduler_type "LinearLR" \
--batch_size 8 \
--num_epochs 15 \
--num_samples 123287 \
--ls_epsilon 0.0 \
--milestone_steps 3 5 7 9 11 13 \
--encoder_out 'img' 'ques' \
--dropout 0.1 \
--img_has_bboxes \
--ca_has_layer_norm \
--ca_has_updated_hist \
--ca_num_cross_attns 2 \
--ca_every_head_attn \
--ca_has_residual \
--ca_has_intra_attns \
--txt_has_layer_norm \
--txt_has_decoder_layer_norm \
--txt_has_pos_embedding \
--val_feat_img_path "~/datasets/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_50_50.h5" \
--train_feat_img_path "~/datasets/bottom-up-attention/trainval_resnet101_faster_rcnn_genome__num_boxes_50_50.h5" \
>> /home/quang/workspace/log/qsub/train_simple/n49.txt