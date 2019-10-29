#### FOR S10 ######
# cd /home/quang/workspace/repos/visdial/scripts/train_simple/finetune/s08_ft.sh
CUDA_VISIBLE_DEVICES=0 python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 10 \
--config '/home/quang/workspace/repos/visdial/configs/train_simple_s10_simple_branch.yml' \
--path_pretrained_ckpt '/media/local_workspace/quang/checkpoints/visdial/CVPR/train_simple/lr001/12_epochs/s10_simple_branch/checkpoint_11.pth'

# cd /home/quang/workspace/repos/visdial/scripts/train_simple/finetune/s08_ft.sh
CUDA_VISIBLE_DEVICES=0 python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 2e-5 \
--num_epochs 10 \
--config '/home/quang/workspace/repos/visdial/configs/train_simple_s10_simple_branch.yml' \
--path_pretrained_ckpt '/media/local_workspace/quang/checkpoints/visdial/CVPR/train_simple/lr001/12_epochs/s10_simple_branch/checkpoint_11.pth'

# cd /home/quang/workspace/repos/visdial/scripts/train_simple/finetune/s08_ft.sh
CUDA_VISIBLE_DEVICES=0 python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 1e-5 \
--num_epochs 10 \
--config '/home/quang/workspace/repos/visdial/configs/train_simple_s10_simple_branch.yml' \
--path_pretrained_ckpt '/media/local_workspace/quang/checkpoints/visdial/CVPR/train_simple/lr001/12_epochs/s10_simple_branch/checkpoint_11.pth'