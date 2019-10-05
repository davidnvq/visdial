CUDA_VISIBLE_DEVICES=2,3 python /home/quang/workspace/repos/visdial/finetune.py \
--num_epochs 10 \
--config '/home/quang/workspace/repos/visdial/configs/v006_ABC_LP_lkf_D25.yml' \
--path_pretrained_ckpt '/media/local_workspace/quang/checkpoints/visdial/CVPR/v006_ABC_LP_lkf_D25/checkpoint_last.pth'
