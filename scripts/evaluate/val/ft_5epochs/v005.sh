CUDA_VISIBLE_DEVICES=0 python /home/quang/workspace/repos/visdial/evaluate.py \
--config "/home/quang/workspace/repos/visdial/configs/v005_ABC_LP_Lkf_D36.yml" \
--weights /media/local_workspace/quang/checkpoints/visdial/CVPR/v005_ABC_LP_Lkf_D36/finetune/checkpoint_5.pth \
--split "val" \
--decoder_type 'disc' \
--save_ranks_path '/home/quang/ranks/ft_5epochs/val/v005_ABC_LP_Lkf_D36_disc.json' \
--output_path '/home/quang/ranks/ft_5epochs/val/v005_ABC_LP_Lkf_D36_disc.pkl'