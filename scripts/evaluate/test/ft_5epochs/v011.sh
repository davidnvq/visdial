CUDA_VISIBLE_DEVICES=1 python /home/quang/workspace/repos/visdial/evaluate.py \
--config "/home/quang/workspace/repos/visdial/configs/v011_ABC_LP_lkf_D36.yml" \
--weights /media/local_workspace/quang/checkpoints/visdial/CVPR/v011_ABC_LP_lkf_D36/finetune/checkpoint_5.pth \
--split "test" \
--decoder_type 'disc' \
--save_ranks_path '/home/quang/ranks/ft_5epochs/test/v011_ABC_LP_lkf_D36_disc.json' \
--output_path '/home/quang/ranks/ft_5epochs/test/v011_ABC_LP_lkf_D36_disc.pkl'
