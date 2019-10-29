/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--wpath '/home/quang/checkpoints/s13/finetune/lr_5e-05/CosineLR/checkpoint_3.pth' \
--cpath '/home/quang/checkpoints/s13/config.json' \
--split "test" \
--decoder_type 'disc' \
--ckpt_name 'ft_ckpt_3' \
--device 'cuda:0'

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--wpath '/home/quang/checkpoints/s15/finetune/lr_5e-05/CosineLR/checkpoint_3.pth' \
--cpath '/home/quang/checkpoints/s15/config.json' \
--split "test" \
--decoder_type 'disc' \
--ckpt_name 'ft_ckpt_3' \
--device 'cuda:0'