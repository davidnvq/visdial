CUDA_VISIBLE_DEVICES=2,3 python /home/quang/workspace/repos/visdial/finetune.py \
--num_epochs 10 \
--config '/home/quang/workspace/repos/visdial/configs/v011_ABC_LP_lkf_D36.yml' \
--path_pretrained_ckpt '/media/local_workspace/quang/checkpoints/visdial/CVPR/v011_ABC_LP_lkf_D50/checkpoint_last.pth'
