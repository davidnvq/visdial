python evaluate.py \
--cpath /home/quang/checkpoints/abci/s44/config.json \
--wpath /home/quang/checkpoints/abci/s44/finetune/lr_1e-05/CosineLR/checkpoint_0.pth \
--split "test" \
--ckpt_name "ft_ckpt_0" \
--device "cuda:1"