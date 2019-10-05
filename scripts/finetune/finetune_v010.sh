CUDA_VISIBLE_DEVICES=0,1 python /home/quang/workspace/repos/visdial/finetune.py \
--num_epochs 10 \
--config '/home/quang/workspace/repos/visdial/configs/v010_ABC_LP_Lkf_DAda.yml' \
--path_pretrained_ckpt '/media/local_workspace/quang/checkpoints/visdial/CVPR/v010_ABC_LP_Lkf_DAda/checkpoint_last.pth'
