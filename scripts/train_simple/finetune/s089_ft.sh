#### FOR S08 ######
# cd /home/quang/workspace/repos/visdial/scripts/train_simple/finetune/s08_ft.sh
CUDA_VISIBLE_DEVICES=5 python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 5e-5 \
--num_epochs 10 \
--config '/home/quang/workspace/repos/visdial/configs/train_simple_s08_simple_branch.yml' \
--path_pretrained_ckpt '/media/local_workspace/quang/checkpoints/visdial/CVPR/train_simple/lr001/s08_simple_branch/checkpoint_11.pth'

#### FOR S08 ######
# cd /home/quang/workspace/repos/visdial/scripts/train_simple/finetune/s08_ft.sh
CUDA_VISIBLE_DEVICES=5 python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 2e-5 \
--num_epochs 10 \
--config '/home/quang/workspace/repos/visdial/configs/train_simple_s08_simple_branch.yml' \
--path_pretrained_ckpt '/media/local_workspace/quang/checkpoints/visdial/CVPR/train_simple/lr001/s08_simple_branch/checkpoint_11.pth'


CUDA_VISIBLE_DEVICES=5 python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 2e-5 \
--num_epochs 10 \
--config '/home/quang/workspace/repos/visdial/configs/train_simple_s09_simple_branch.yml' \
--path_pretrained_ckpt '/media/local_workspace/quang/checkpoints/visdial/CVPR/train_simple/lr001/s09_simple_branch/checkpoint_last.pth'

CUDA_VISIBLE_DEVICES=5 python /home/quang/workspace/repos/visdial/finetune.py \
--batch_size 12 \
--init_lr 1e-5 \
--num_epochs 10 \
--config '/home/quang/workspace/repos/visdial/configs/train_simple_s09_simple_branch.yml' \
--path_pretrained_ckpt '/media/local_workspace/quang/checkpoints/visdial/CVPR/train_simple/lr001/s09_simple_branch/checkpoint_last.pth'

