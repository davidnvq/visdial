#!/usr/bin/env sh
#$-pe gpu 4
#$-l gpu=4
#$-j y
#$-cwd
#$-V
#$-o /home/quang/workspace/log/qsub/v010_ABC_LP_Lkf_DAda.log
#$-q main.q@yagi02.vision.is.tohoku

export CUDA_VISIBLE_DEVICES=$SGE_GPU
python /home/quang/workspace/repos/visdial/train.py \
--config_name v010_ABC_LP_Lkf_DAda \
--decoder_type misc \
--init_lr 0.005 \
--batch_size 8 \
--num_epochs 30 \
--num_samples 123287 \
--ls_epsilon 0.1 \
--dropout 0.1 \
--ca_has_layer_norm \
--ca_has_shared_attns \
--ca_num_cross_attns 1 \
--img_has_bboxes \
--img_has_classes \
--img_has_attributes \
--txt_has_layer_norm \
--txt_has_decoder_layer_norm \
--txt_has_pos_embedding \
--val_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_(10, 100).h5" \
--train_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/trainval_resnet101_faster_rcnn_genome__num_boxes_(10, 100).h5" \
>> /home/quang/workspace/log/qsub/v010_ABC_LP_Lkf_DAda.txt