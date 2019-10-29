CUDA_VISIBLE_DEVICES=1 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--wpath '/home/quang/checkpoints/s25/checkpoint_5.pth' \
--cpath '/home/quang/checkpoints/s25/config.json' \
--split "test" \
--decoder_type 'disc' \
--ckpt_name 'no_ft_ckpt_5' \
--device 'cuda:1'
