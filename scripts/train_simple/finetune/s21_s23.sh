CUDA_VISIBLE_DEVICES=0,1 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 10 \
--config '/home/quang/checkpoints/s17/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/s17/checkpoint_11.pth'

CUDA_VISIBLE_DEVICES=0,1 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 10 \
--config '/home/quang/checkpoints/s18/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/s18/checkpoint_11.pth'

CUDA_VISIBLE_DEVICES=0,1 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 10 \
--config '/home/quang/checkpoints/s21/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/s21/checkpoint_11.pth'

CUDA_VISIBLE_DEVICES=0,1 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 10 \
--config '/home/quang/checkpoints/s26/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/s26/checkpoint_11.pth'

