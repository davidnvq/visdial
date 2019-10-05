CUDA_VISIBLE_DEVICES=0,1 python /home/quang/workspace/repos/visdial/finetune.py \
--num_epochs 10 \
--config '/home/quang/workspace/repos/visdial/configs/v005_ABC_LP_Lkf_D36.yml' \
--path_pretrained_ckpt '/media/local_workspace/quang/checkpoints/visdial/CVPR/v005_ABC_LP_Lkf_D36/checkpoint_last.pth'
