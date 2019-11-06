### LR 1e-5
CUDA_VISIBLE_DEVICES=2,3 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 1e-5 \
--num_epochs 5 \
--config '/home/quang/checkpoints/abci/s45/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/abci/s45/checkpoint_14.pth'

CUDA_VISIBLE_DEVICES=2,3 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 1e-5 \
--num_epochs 10 \
--config '/home/quang/checkpoints/abci/n41/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/abci/n41/checkpoint_14.pth'


########################## Evaluate
/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/ \
--cpath /home/quang/checkpoints/abci/s45/config.json \
--wpath /home/quang/checkpoints/abci/s45/finetune/lr_1e-05/CosineLR/checkpoint_0.pth \
--split "test" \
--ckpt_name "ft_ckpt_0" \
--device "cuda:2"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/ \
--cpath /home/quang/checkpoints/abci/s45/config.json \
--wpath /home/quang/checkpoints/abci/s45/finetune/lr_1e-05/CosineLR/checkpoint_1.pth \
--split "test" \
--ckpt_name "ft_ckpt_1" \
--device "cuda:2"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/ \
--cpath /home/quang/checkpoints/abci/n41/config.json \
--wpath /home/quang/checkpoints/abci/n41/finetune/lr_1e-05/CosineLR/checkpoint_0.pth \
--split "test" \
--ckpt_name "ft_ckpt_0" \
--device "cuda:2"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/ \
--cpath /home/quang/checkpoints/abci/n41/config.json \
--wpath /home/quang/checkpoints/abci/n41/finetune/lr_1e-05/CosineLR/checkpoint_1.pth \
--split "test" \
--ckpt_name "ft_ckpt_1" \
--device "cuda:2"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/ \
--cpath /home/quang/checkpoints/abci/n42/config.json \
--wpath /home/quang/checkpoints/abci/n42/finetune/lr_1e-05/CosineLR/checkpoint_0.pth \
--split "test" \
--ckpt_name "ft_ckpt_0" \
--device "cuda:2"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/ \
--cpath /home/quang/checkpoints/abci/n42/config.json \
--wpath /home/quang/checkpoints/abci/n42/finetune/lr_1e-05/CosineLR/checkpoint_1.pth \
--split "test" \
--ckpt_name "ft_ckpt_1" \
--device "cuda:2"
########################## LR 5e-5
CUDA_VISIBLE_DEVICES=2,3 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 5 \
--config '/home/quang/checkpoints/abci/s41/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/abci/s41/checkpoint_13.pth'

CUDA_VISIBLE_DEVICES=2,3 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 5 \
--config '/home/quang/checkpoints/abci/n41/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/abci/n41/checkpoint_14.pth'

CUDA_VISIBLE_DEVICES=2,3 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 5 \
--config '/home/quang/checkpoints/abci/n42/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/abci/n42/checkpoint_14.pth'

########################## Evaluate
/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/ \
--cpath /home/quang/checkpoints/abci/s41/config.json \
--wpath /home/quang/checkpoints/abci/s41/finetune/lr_5e-05/CosineLR/checkpoint_4.pth \
--split "test" \
--ckpt_name "ft_ckpt_4" \
--device "cuda:2"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/ \
--cpath /home/quang/checkpoints/abci/s41/config.json \
--wpath /home/quang/checkpoints/abci/s41/finetune/lr_5e-05/CosineLR/checkpoint_5.pth \
--split "test" \
--ckpt_name "ft_ckpt_5" \
--device "cuda:2"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/ \
--cpath /home/quang/checkpoints/abci/n41/config.json \
--wpath /home/quang/checkpoints/abci/n41/finetune/lr_5e-05/CosineLR/checkpoint_4.pth \
--split "test" \
--ckpt_name "ft_ckpt_4" \
--device "cuda:2"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/ \
--cpath /home/quang/checkpoints/abci/n41/config.json \
--wpath /home/quang/checkpoints/abci/n41/finetune/lr_5e-05/CosineLR/checkpoint_5.pth \
--split "test" \
--ckpt_name "ft_ckpt_5" \
--device "cuda:2"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/ \
--cpath /home/quang/checkpoints/abci/n42/config.json \
--wpath /home/quang/checkpoints/abci/n42/finetune/lr_5e-05/CosineLR/checkpoint_4.pth \
--split "test" \
--ckpt_name "ft_ckpt_4" \
--device "cuda:2"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/ \
--cpath /home/quang/checkpoints/abci/n42/config.json \
--wpath /home/quang/checkpoints/abci/n42/finetune/lr_5e-05/CosineLR/checkpoint_5.pth \
--split "test" \
--ckpt_name "ft_ckpt_5" \
--device "cuda:2"