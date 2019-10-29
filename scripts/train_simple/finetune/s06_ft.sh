CUDA_VISIBLE_DEVICES=6,7 python /home/administrator/quang/workspace/repos/visdial/finetune.py \
--num_epochs 15 \
--config '/home/administrator/quang/workspace/repos/visdial/configs/train_simple_s06_simple_branch.yml' \
--path_pretrained_ckpt '/media/local_workspace/quang/checkpoints/visdial/CVPR/train_simple/lr001/s06_simple_branch/checkpoint_12.pth'
