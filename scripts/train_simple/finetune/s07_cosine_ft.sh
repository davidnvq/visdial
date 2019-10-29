CUDA_VISIBLE_DEVICES=4,5 python /home/administrator/quang/workspace/repos/visdial/finetune.py \
--num_epochs 15 \
--config '/home/administrator/quang/workspace/repos/visdial/configs/train_simple_s07_simple_branch_cosine.yml' \
--path_pretrained_ckpt '/media/local_workspace/quang/checkpoints/visdial/CVPR/train_simple/lr001/s07_simple_branch_cosine/checkpoint_7.pth'