#! /bin/bash

rsync -av \
--exclude=.git \
--exclude=*pyc \
--exclude=*idea \
--exclude=*ignore \
--exclude=*.ipynb \
--exclude=*DS_Store \
--exclude=__pycache__ \
--exclude=*ipynb_checkpoints \
../visdial quang@k2:/home/quang/repos