/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/train.py \
--config_name overfit/fp16/v003_aBc_LP_Lkf_D36 \
--decoder_type misc \
--init_lr 0.005 \
--batch_size 8 \
--num_epochs 30 \
--num_samples 2064 \
--ls_epsilon 0.1 \
--dropout 0.1 \
--ca_has_layer_norm \
--ca_has_shared_attns \
--ca_num_cross_attns 1 \
--img_has_classes \
--txt_has_layer_norm \
--txt_has_decoder_layer_norm \
--txt_has_pos_embedding \
--val_feat_img_path "/home/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_(36, 36).h5" \
--train_feat_img_path "/home/quang/datasets/visdial/bottom-up-attention/val2018_resnet101_faster_rcnn_genome__num_boxes_(36, 36).h5" \
--train_json_dialog_path "/home/quang/datasets/visdial/annotations/visdial_1.0_val.json"
# >> /home/administrator/quang/workspace/log/fp16/v003_aBc_LP_Lkf_D36.txt

