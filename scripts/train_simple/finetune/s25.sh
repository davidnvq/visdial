CUDA_VISIBLE_DEVICES=1 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 10 \
--config '/home/quang/checkpoints/s25/config.json' \
--path_pretrained_ckpt '/home/quang/checkpoints/s25/checkpoint_13.pth'