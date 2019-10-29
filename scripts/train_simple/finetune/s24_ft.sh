CUDA_VISIBLE_DEVICES=2,3 /home/quang/anaconda3/bin/python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 10 \
--config '/media/local_workspace/quang/checkpoints/visdial/CVPR/train_simple/s24/train_simple_12epochs_s24.yml' \
--path_pretrained_ckpt '/media/local_workspace/quang/checkpoints/visdial/CVPR/train_simple/s24/checkpoint_11.pth'