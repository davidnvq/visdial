CUDA_VISIBLE_DEVICES=1 python /home/quang/workspace/repos/visdial/evaluate.py \
--config "/home/quang/workspace/repos/visdial/configs/v007_ABC_LP_lkf_D36_ca2.yml" \
--weights /media/local_workspace/quang/checkpoints/visdial/CVPR/v007_ABC_LP_lkf_D36_ca2/finetune/checkpoint_5.pth \
--split "test" \
--decoder_type 'disc' \
--save_ranks_path '/home/quang/ranks/ft_5epochs/test/v007_ABC_LP_lkf_D36_ca2.json' \
--output_path '/home/quang/ranks/ft_5epochs/test/v007_ABC_LP_lkf_D36_ca2.pkl'