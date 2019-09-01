#!/usr/bin/env bash
python train.py \
--config_name overfit/misc_dlegacy \
--decoder_type misc \
--init_lr 0.001 \
--num_epochs 2 \
--overfit \
--num_samples 32 \
--ls_epsilon 0.1 \
--dropout 0.1 \
--ca_has_layer_norm \
--ca_has_shared_attns \
--ca_num_cross_attns 1 \
--txt_has_layer_norm \
--txt_has_decoder_layer_norm \
--txt_has_pos_embedding \
--val_feat_img_path "/media/local_workspace/quang/datasets/visdial/legacy/features_faster_rcnn_x101_val.h5" \
--train_feat_img_path "/media/local_workspace/quang/datasets/visdial/legacy/features_faster_rcnn_x101_val.h5" \
--train_json_dialog_path "/media/local_workspace/quang/datasets/visdial/annotations/visdial_1.0_val.json"