# LF Discriminative on validation set.
#python evaluate.py \
#--split 'val' \
#--gpu-ids 0 1 \
#--cpu-workers 4 \
#--config-yml 'configs/lf_disc_faster_rcnn_x101.yml' \
#--val-json '/home/ubuntu/datasets/myvisdial/data/visdial_1.0_val.json' \
#--val-dense-json '/home/ubuntu/datasets/myvisdial/data/visdial_1.0_val_dense_annotations.json' \
#--load-pthpath '/home/ubuntu/datasets/myvisdial/checkpoints/lf_disc/baseline/lf_disc_faster_rcnn_x101_trainval.pth' \
#--save-ranks-path '/home/ubuntu/datasets/myvisdial/checkpoints/lf_disc/baseline/output/val_ranks.json'


# LF Generative on validation set.
python evaluate.py \
--split 'val' \
--gpu-ids 0 1 \
--cpu-workers 4 \
--config-yml 'configs/lf_gen_faster_rcnn_x101.yml' \
--val-json '/home/ubuntu/datasets/myvisdial/data/visdial_1.0_val.json' \
--val-dense-json '/home/ubuntu/datasets/myvisdial/data/visdial_1.0_val_dense_annotations.json' \
--load-pthpath '/home/ubuntu/datasets/myvisdial/checkpoints/gen_disc/baseline/lf_gen_faster_rcnn_x101_train.pth' \
--save-ranks-path '/home/ubuntu/datasets/myvisdial/checkpoints/lf_gen/baseline/output/val_ranks.json'
