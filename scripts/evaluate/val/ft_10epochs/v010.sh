CUDA_VISIBLE_DEVICES=1 python /home/quang/workspace/repos/visdial/evaluate.py \
--config "/home/quang/workspace/repos/visdial/configs/v010_ABC_LP_Lkf_DAda.yml" \
--weights /media/local_workspace/quang/checkpoints/visdial/CVPR/v010_ABC_LP_Lkf_DAda/finetune/checkpoint_last.pth \
--split "val" \
--decoder_type 'disc' \
--save_ranks_path '/home/quang/ranks/ft_10epochs/val/v010_ABC_LP_Lkf_DAda_disc.json' \
--output_path '/home/quang/ranks/ft_10epochs/val/v010_ABC_LP_Lkf_DAda_disc.pkl'
