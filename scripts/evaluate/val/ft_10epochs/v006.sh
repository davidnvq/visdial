CUDA_VISIBLE_DEVICES=0 python /home/quang/workspace/repos/visdial/evaluate.py \
--config "/home/quang/workspace/repos/visdial/configs/v006_ABC_LP_lkf_D25.yml" \
--weights /media/local_workspace/quang/checkpoints/visdial/CVPR/v006_ABC_LP_lkf_D25/finetune/checkpoint_last.pth \
--split "val" \
--decoder_type 'disc' \
--save_ranks_path '/home/quang/ranks/ft_10epochs/val/v006_ABC_LP_lkf_D25_disc.json' \
--output_path '/home/quang/ranks/ft_10epochs/val/v006_ABC_LP_lkf_D25_disc.pkl'
