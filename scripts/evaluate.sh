# cd ~/Dropbox/repos/visdial/; bash evaluate_lf_baselines.sh

############################################################
# LF Discriminative on validation set.
#############################################################
#CUDA_VISIBLE_DEVICES='1' python evaluate.py \
#--split 'val' \
#--gpu-ids 1 \
#--cpu-workers 4 \
#--config-yml 'configs/lf_disc_faster_rcnn_x101.yml' \
#--val-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_val.json' \
#--val-dense-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_val_dense_annotations.json' \
#--load-pthpath '/home/ubuntu/datasets/visdial/checkpoints/lf_disc/baseline/lf_disc_faster_rcnn_x101_trainval.pth' \
#--save-ranks-path '/home/ubuntu/datasets/visdial/checkpoints/lf_disc/reproduced/submission/disc_val_ranks.json'

############################################################
# LF Generative on validation set.
############################################################

## For validation
#CUDA_VISIBLE_DEVICES='1' python evaluate.py \
#--split 'val' \
#--gpu-ids 1 \
#--cpu-workers 4 \
#--config-yml 'configs/lf_gen_faster_rcnn_x101.yml' \
#--val-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_val.json' \
#--val-dense-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_val_dense_annotations.json' \
#--load-pthpath '/home/ubuntu/datasets/visdial/checkpoints/lf_gen_bilstm/may10/checkpoint_9.pth' \
#--save-ranks-path '/home/ubuntu/datasets/visdial/checkpoints/lf_gen_bilstm/may10/submission/bilstm_gen_val_ranks.json' \


## For testing
#CUDA_VISIBLE_DEVICES='1' python evaluate.py \
#--split 'test' \
#--gpu-ids 1 \
#--cpu-workers 4 \
#--config-yml 'configs/lf_gen_faster_rcnn_x101.yml' \
#--load-pthpath '/home/ubuntu/datasets/visdial/checkpoints/lf_gen/may8/checkpoint_9.pth' \
#--save-ranks-path '/home/ubuntu/datasets/visdial/checkpoints/lf_gen/may8/submission/test_ranks.json' \
#--test-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_test.json'
#


############################################################
# BI-LSTM LF Generative on validation set.
############################################################

## cd ~/Dropbox/repos/visdial/; bash evaluate_lf_baselines.sh
## For validation
#CUDA_VISIBLE_DEVICES='1' python evaluate.py \
#--split 'val' \
#--gpu-ids 1 \
#--cpu-workers 4 \
#--config-yml 'configs/lf_gen_faster_rcnn_x101.yml' \
#--val-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_val.json' \
#--val-dense-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_val_dense_annotations.json' \
#--load-pthpath '/home/ubuntu/datasets/visdial/checkpoints/lf_gen_bilstm/may10/checkpoint_9.pth' \
#--save-ranks-path '/home/ubuntu/datasets/visdial/checkpoints/lf_gen/may8/submission/val_ranks.json' \


## For testing
#CUDA_VISIBLE_DEVICES='1' python evaluate.py \
#--split 'test' \
#--gpu-ids 1 \
#--cpu-workers 4 \
#--config-yml 'configs/lf_gen_faster_rcnn_x101.yml' \
#--load-pthpath '/home/ubuntu/datasets/visdial/checkpoints/lf_gen/may8/checkpoint_9.pth' \
#--save-ranks-path '/home/ubuntu/datasets/visdial/checkpoints/lf_gen/may8/submission/test_ranks.json' \
#--test-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_test.json'


############################################################
# ENSEMBLE
############################################################
CUDA_VISIBLE_DEVICES='1' python ensemble.py \
--split 'val' \
--gpu-ids 1 \
--cpu-workers 4 \
--config-ymls 'configs/lf_disc_faster_rcnn_x101.yml' 'configs/lf_gen_faster_rcnn_x101.yml' \
--val-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_val.json' \
--val-dense-json '/home/ubuntu/datasets/visdial/data/visdial_1.0_val_dense_annotations.json' \
--load-pthpaths \
'/home/ubuntu/datasets/visdial/checkpoints/lf_disc/baseline/lf_disc_faster_rcnn_x101_trainval.pth' \
'/home/ubuntu/datasets/visdial/checkpoints/lf_gen/may8/checkpoint_9.pth' \
--save-ranks-path '/home/ubuntu/datasets/visdial/checkpoints/lf_disc/reproduced/submission/disc_val_ranks.json'
