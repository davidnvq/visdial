rsync -av \
--update \
--exclude=.git \
--exclude=*pyc \
--exclude=*idea \
--exclude=*ignore \
--exclude=*DS_Store \
--exclude=__pycache__ \
--exclude=*ipynb_checkpoints \
/media/local_workspace/quang/datasets/visdial/ quang@yagi08:/media/local_workspace/quang/datasets/

# quang@yagi14:/media/local_workspace/quang/checkpoints/visdial/CVPR /media/local_workspace/quang/checkpoints/visdial/