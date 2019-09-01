python /home/quang/workspace/repos/visdial/train.py \
--config_name overfit/v001_abc_LP_lkf_Dlegacy \
--decoder_type misc \
--init_lr 0.005 \
--batch_size 24 \
--num_epochs 50 \
--num_samples 2064 \
--ls_epsilon 0.0 \
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
>> /home/quang/workspace/log/overfit/v001_abc_LP_lkf_Dlegacy.txt

python /home/quang/workspace/repos/visdial/train.py \
--config_name overfit/v002_abc_LP_lkf_D36 \
--decoder_type misc \
--init_lr 0.005 \
--batch_size 24 \
--num_epochs 50 \
--num_samples 2064 \
--ls_epsilon 0.0 \
--dropout 0.1 \
--ca_has_layer_norm \
--ca_has_shared_attns \
--ca_num_cross_attns 1 \
--txt_has_layer_norm \
--txt_has_decoder_layer_norm \
--txt_has_pos_embedding \
--val_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_(36, 36).h5" \
--train_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/trainval_resnet101_faster_rcnn_genome__num_boxes_(36, 36).h5" \
--train_json_dialog_path "/media/local_workspace/quang/datasets/visdial/annotations/visdial_1.0_val.json"
>> /home/quang/workspace/log/overfit/v002_abc_LP_lkf_D36.txt

python /home/quang/workspace/repos/visdial/train.py \
--config_name overfit/v003_aBc_LP_Lkf_D36 \
--decoder_type misc \
--init_lr 0.005 \
--batch_size 24 \
--num_epochs 50 \
--num_samples 2064 \
--ls_epsilon 0.1 \
--dropout 0.1 \
--ca_has_layer_norm \
--ca_has_shared_attns \
--ca_num_cross_attns 1 \
--img_has_bboxes \
--txt_has_layer_norm \
--txt_has_decoder_layer_norm \
--txt_has_pos_embedding \
--val_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_(36, 36).h5" \
--train_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/trainval_resnet101_faster_rcnn_genome__num_boxes_(36, 36).h5" \
--train_json_dialog_path "/media/local_workspace/quang/datasets/visdial/annotations/visdial_1.0_val.json"
>> /home/quang/workspace/log/overfit/v003_aBc_LP_Lkf_D36.txt

python /home/quang/workspace/repos/visdial/train.py \
--config_name overfit/v004_aBC_LP_Lkf_D36 \
--decoder_type misc \
--init_lr 0.005 \
--batch_size 24 \
--num_epochs 50 \
--num_samples 2064 \
--ls_epsilon 0.1 \
--dropout 0.1 \
--ca_has_layer_norm \
--ca_has_shared_attns \
--ca_num_cross_attns 1 \
--img_has_bboxes \
--img_has_classes \
--txt_has_layer_norm \
--txt_has_decoder_layer_norm \
--txt_has_pos_embedding \
--val_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_(36, 36).h5" \
--train_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_(36, 36).h5" \
--train_json_dialog_path "/media/local_workspace/quang/datasets/visdial/annotations/visdial_1.0_val.json"
>> /home/quang/workspace/log/overfit/v004_aBC_LP_Lkf_D36.txt


python /home/quang/workspace/repos/visdial/train.py \
--config_name overfit/v005_ABC_LP_Lkf_D36 \
--decoder_type misc \
--init_lr 0.005 \
--batch_size 24 \
--num_epochs 50 \
--num_samples 2064 \
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
--val_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_(36, 36).h5" \
--train_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_(36, 36).h5" \
--train_json_dialog_path "/media/local_workspace/quang/datasets/visdial/annotations/visdial_1.0_val.json"
>> /home/quang/workspace/log/overfit/v005_ABC_LP_Lkf_D36.txt

python /home/quang/workspace/repos/visdial/train.py \
--config_name overfit/v005_ABC_LP_Lkf_D25 \
--decoder_type misc \
--init_lr 0.005 \
--batch_size 24 \
--num_epochs 50 \
--num_samples 2064 \
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
--val_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_(25, 25).h5" \
--train_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_(25, 25).h5" \
--train_json_dialog_path "/media/local_workspace/quang/datasets/visdial/annotations/visdial_1.0_val.json"
>> /home/quang/workspace/log/overfit/v005_ABC_LP_Lkf_D25.txt

python /home/quang/workspace/repos/visdial/train.py \
--config_name overfit/v007_ABC_LP_Lkf_D36_ca2 \
--decoder_type misc \
--init_lr 0.005 \
--batch_size 24 \
--num_epochs 50 \
--num_samples 2064 \
--ls_epsilon 0.1 \
--dropout 0.1 \
--ca_has_layer_norm \
--ca_has_shared_attns \
--ca_num_cross_attns 2 \
--img_has_bboxes \
--img_has_classes \
--img_has_attributes \
--txt_has_layer_norm \
--txt_has_decoder_layer_norm \
--txt_has_pos_embedding \
--val_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_(36, 36).h5" \
--train_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_(36, 36).h5" \
--train_json_dialog_path "/media/local_workspace/quang/datasets/visdial/annotations/visdial_1.0_val.json"
>> /home/quang/workspace/log/overfit/v007_ABC_LP_Lkf_D36_ca2.txt

python /home/quang/workspace/repos/visdial/train.py \
--config_name overfit/v008_ABC_LP_Lkf_D36_ca2res \
--decoder_type misc \
--init_lr 0.005 \
--batch_size 24 \
--num_epochs 50 \
--num_samples 2064 \
--ls_epsilon 0.1 \
--dropout 0.1 \
--ca_has_layer_norm \
--ca_has_shared_attns \
--ca_has_residual \
--ca_num_cross_attns 2 \
--img_has_bboxes \
--img_has_classes \
--img_has_attributes \
--txt_has_layer_norm \
--txt_has_decoder_layer_norm \
--txt_has_pos_embedding \
--val_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_(36, 36).h5" \
--train_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_(36, 36).h5" \
--train_json_dialog_path "/media/local_workspace/quang/datasets/visdial/annotations/visdial_1.0_val.json"
>> /home/quang/workspace/log/overfit/v008_ABC_LP_Lkf_D36_ca2res.txt

python /home/quang/workspace/repos/visdial/train.py \
--config_name overfit/v009_ABC_LP_Lkf_D50 \
--decoder_type misc \
--init_lr 0.005 \
--batch_size 24 \
--num_epochs 50 \
--num_samples 2064 \
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
--val_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_(50, 50).h5" \
--train_feat_img_path "/media/local_workspace/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_(50, 50).h5" \
--train_json_dialog_path "/media/local_workspace/quang/datasets/visdial/annotations/visdial_1.0_val.json"
>> /home/quang/workspace/log/overfit/v009_ABC_LP_Lkf_D50.txt
