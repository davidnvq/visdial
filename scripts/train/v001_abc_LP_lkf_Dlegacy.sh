#!/usr/bin/env sh
#$-pe gpu 4
#$-l gpu=4
#$-j y
#$-cwd
#$-V
#$-o /home/quang/workspace/log/qsub/v001_abc_LP_lkf_Dlegacy.log
#$-q main.q@yagi14.vision.is.tohoku

export CUDA_VISIBLE_DEVICES=$SGE_GPU
python /home/quang/workspace/repos/visdial/train.py \
--config_name test/v001_abc_LP_lkf_Dlegacy \
--decoder_type misc \
--init_lr 0.005 \
--batch_size 8 \
--num_epochs 30 \
--num_samples 123287 \
--ls_epsilon 0.0 \
--dropout 0.1 \
--ca_has_layer_norm \
--ca_has_shared_attns \
--ca_num_cross_attns 1 \
--txt_has_layer_norm \
--txt_has_decoder_layer_norm \
--txt_has_pos_embedding \
--val_feat_img_path "/media/local_workspace/quang/datasets/visdial/legacy/features_faster_rcnn_x101_val.h5" \
--train_feat_img_path "/media/local_workspace/quang/datasets/visdial/legacy/features_faster_rcnn_x101_train.h5" \
>> /home/quang/workspace/log/qsub/v001_abc_LP_lkf_Dlegacy.txt