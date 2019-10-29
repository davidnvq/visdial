
CUDA_VISIBLE_DEVICES=2,3 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 10 \
--config '/home/quang/checkpoints/s12/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/s12/checkpoint_11.pth'

CUDA_VISIBLE_DEVICES=2,3 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 10 \
--config '/home/quang/checkpoints/s14/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/s14/checkpoint_11.pth'

CUDA_VISIBLE_DEVICES=2,3 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 10 \
--config '/home/quang/checkpoints/s15/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/s15/checkpoint_11.pth'

CUDA_VISIBLE_DEVICES=2,3 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 10 \
--config '/home/quang/checkpoints/s16/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/s16/checkpoint_11.pth'