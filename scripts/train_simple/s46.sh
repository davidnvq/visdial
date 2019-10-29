#!/usr/bin/env sh
#$-pe gpu 4
#$-l gpu=4
#$-j y
#$-cwd
#$-V
#$-o /home/quang/workspace/log/qsub/train_simple/s46.log
#$-q main.q@yagi07.vision.is.tohoku

export CUDA_VISIBLE_DEVICES=$SGE_GPU
python /home/quang/workspace/repos/visdial/train.py \
--config_name s46 \
--decoder_type misc \
--init_lr 0.001 \
--scheduler_type "LinearLR" \
--batch_size 8 \
--num_epochs 15 \
--num_samples 123287 \
--ls_epsilon 0.0 \
--milestone_steps 3 5 7 9 11 13 \
--encoder_out 'img' 'ques' 'hist' \
--dropout 0.1 \
--img_has_bboxes \
--ca_has_layer_norm \
--ca_num_cross_attns 1 \
--ca_has_residual \
--ca_has_intra_attns \
--ca_has_updated_hist \
--txt_has_layer_norm \
--txt_has_nsl \
--txt_has_decoder_layer_norm \
--txt_has_pos_embedding \
--val_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_100_100.h5" \
--train_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/trainval_resnet101_faster_rcnn_genome__num_boxes_100_100.h5" \
>> /home/quang/workspace/log/qsub/train_simple/s46.txt