### LR 1e-5
CUDA_VISIBLE_DEVICES=0,1 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 1e-5 \
--num_epochs 10 \
--config '/home/quang/checkpoints/n45/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/n45/checkpoint_13.pth'

### LR 5e-5
CUDA_VISIBLE_DEVICES=0,1 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 10 \
--config '/home/quang/checkpoints/n45/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/n45/checkpoint_13.pth'

### LR 1e-5
CUDA_VISIBLE_DEVICES=0,1 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 1e-5 \
--num_epochs 10 \
--config '/home/quang/checkpoints/n46/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/n46/checkpoint_13.pth'
### LR 5e-5
CUDA_VISIBLE_DEVICES=0,1 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 10 \
--config '/home/quang/checkpoints/n46/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/n46/checkpoint_13.pth'


########################## Evaluate
/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n45/config.json \
--wpath /home/quang/checkpoints/n45/finetune/lr_1e-05/CosineLR/checkpoint_0.pth \
--split "test" \
--ckpt_name "ft_ckpt_0" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n45/config.json \
--wpath /home/quang/checkpoints/n45/finetune/lr_1e-05/CosineLR/checkpoint_1.pth \
--split "test" \
--ckpt_name "ft_ckpt_1" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n45/config.json \
--wpath /home/quang/checkpoints/n45/finetune/lr_1e-05/CosineLR/checkpoint_2.pth \
--split "test" \
--ckpt_name "ft_ckpt_2" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n46/config.json \
--wpath /home/quang/checkpoints/n46/finetune/lr_1e-05/CosineLR/checkpoint_0.pth \
--split "test" \
--ckpt_name "ft_ckpt_0" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n46/config.json \
--wpath /home/quang/checkpoints/n46/finetune/lr_1e-05/CosineLR/checkpoint_1.pth \
--split "test" \
--ckpt_name "ft_ckpt_1" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n46/config.json \
--wpath /home/quang/checkpoints/n46/finetune/lr_1e-05/CosineLR/checkpoint_2.pth \
--split "test" \
--ckpt_name "ft_ckpt_2" \
--device "cuda:0"

### Evaluate with lr5e-05
/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n45/config.json \
--wpath /home/quang/checkpoints/n45/finetune/lr_5e-05/CosineLR/checkpoint_0.pth \
--split "test" \
--ckpt_name "ft_ckpt_0" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n45/config.json \
--wpath /home/quang/checkpoints/n45/finetune/lr_5e-05/CosineLR/checkpoint_1.pth \
--split "test" \
--ckpt_name "ft_ckpt_1" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n45/config.json \
--wpath /home/quang/checkpoints/n45/finetune/lr_5e-05/CosineLR/checkpoint_2.pth \
--split "test" \
--ckpt_name "ft_ckpt_2" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n45/config.json \
--wpath /home/quang/checkpoints/n45/finetune/lr_5e-05/CosineLR/checkpoint_3.pth \
--split "test" \
--ckpt_name "ft_ckpt_3" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n45/config.json \
--wpath /home/quang/checkpoints/n45/finetune/lr_5e-05/CosineLR/checkpoint_4.pth \
--split "test" \
--ckpt_name "ft_ckpt_4" \
--device "cuda:0"


/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n46/config.json \
--wpath /home/quang/checkpoints/n46/finetune/lr_5e-05/CosineLR/checkpoint_0.pth \
--split "test" \
--ckpt_name "ft_ckpt_0" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n46/config.json \
--wpath /home/quang/checkpoints/n46/finetune/lr_5e-05/CosineLR/checkpoint_1.pth \
--split "test" \
--ckpt_name "ft_ckpt_1" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n46/config.json \
--wpath /home/quang/checkpoints/n46/finetune/lr_5e-05/CosineLR/checkpoint_2.pth \
--split "test" \
--ckpt_name "ft_ckpt_2" \
--device "cuda:0"


/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n46/config.json \
--wpath /home/quang/checkpoints/n46/finetune/lr_5e-05/CosineLR/checkpoint_3.pth \
--split "test" \
--ckpt_name "ft_ckpt_3" \
--device "cuda:0"

/home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/evaluate.py \
--cpath /home/quang/checkpoints/n46/config.json \
--wpath /home/quang/checkpoints/n46/finetune/lr_5e-05/CosineLR/checkpoint_4.pth \
--split "test" \
--ckpt_name "ft_ckpt_4" \
--device "cuda:0"
