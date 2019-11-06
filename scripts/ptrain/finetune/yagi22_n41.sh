### LR 5e-5
CUDA_VISIBLE_DEVICES=0 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 10 \
--config '/home/quang/checkpoints/n41/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/n41/checkpoint_14.pth'

########################## Evaluate
/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n41/config.json \
--wpath /home/quang/checkpoints/n41/finetune/lr_1e-05/CosineLR/checkpoint_0.pth \
--split "test" \
--ckpt_name "ft_ckpt_0" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n41/config.json \
--wpath /home/quang/checkpoints/n41/finetune/lr_1e-05/CosineLR/checkpoint_1.pth \
--split "test" \
--ckpt_name "ft_ckpt_1" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n41/config.json \
--wpath /home/quang/checkpoints/n41/finetune/lr_1e-05/CosineLR/checkpoint_2.pth \
--split "test" \
--ckpt_name "ft_ckpt_2" \
--device "cuda:0"

########################## Evaluate
/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n41/config.json \
--wpath /home/quang/checkpoints/n41/finetune/lr_5e-05/CosineLR/checkpoint_5.pth \
--split "test" \
--ckpt_name "ft_ckpt_5" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n41/config.json \
--wpath /home/quang/checkpoints/n41/finetune/lr_5e-05/CosineLR/checkpoint_4.pth \
--split "test" \
--ckpt_name "ft_ckpt_4" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n41/config.json \
--wpath /home/quang/checkpoints/n41/finetune/lr_5e-05/CosineLR/checkpoint_3.pth \
--split "test" \
--ckpt_name "ft_ckpt_3" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n41/config.json \
--wpath /home/quang/checkpoints/n41/finetune/lr_5e-05/CosineLR/checkpoint_2.pth \
--split "test" \
--ckpt_name "ft_ckpt_2" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n41/config.json \
--wpath /home/quang/checkpoints/n41/finetune/lr_5e-05/CosineLR/checkpoint_1.pth \
--split "test" \
--ckpt_name "ft_ckpt_1" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n41/config.json \
--wpath /home/quang/checkpoints/n41/finetune/lr_5e-05/CosineLR/checkpoint_0.pth \
--split "test" \
--ckpt_name "ft_ckpt_0" \
--device "cuda:0"