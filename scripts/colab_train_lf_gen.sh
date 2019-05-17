#!/usr/bin/env bash
# To run:
# ! git clone https://github.com/quanguet/visdial
# ! cd /content/visdial;git pull https://github.com/quanguet/visdial
# ! bash /content/visdial/scripts/colab_train_lf_gen.sh

# TODO: Rename ROOT
ROOT='/content'
PATH_PROJ=$ROOT/visdial

python $PATH_PROJ/scripts/colab_train.py \
--gdrive \
--num-epochs 10 \
--lr 1e-4 \
--lr-steps 5 \
--encoder 'lf' \
--decoder 'gen' \
--ckpt-path 'may14' \
--load-pthpath 'checkpoint_old_9.pth'