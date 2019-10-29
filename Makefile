.PHONY: clean, yagi22, pull22, clean, all
CUR_DIR = $(CURDIR)

yagi22: clean
	rsync -av \
	--exclude=.git \
	--exclude=*pyc \
	--exclude=*idea \
	--exclude=*ignore \
	--exclude=*.json \
	--exclude=*.ipynb \
	--exclude=*DS_Store \
	--exclude=__pycache__ \
	--exclude=*ipynb_checkpoints \
	$(CUR_DIR) administrator@yagi22:/home/administrator/quang/workspace/repos


abc: clean
	rsync -av \
	--exclude=*pyc \
	--exclude=*idea \
	--exclude=*ignore \
	--exclude=*.json \
	--exclude=*.ipynb \
	--exclude=*DS_Store \
	--exclude=__pycache__ \
	--exclude=*ipynb_checkpoints \
	$(CUR_DIR) acb11402ci@abc:/home/acb11402ci/workspace/repos

pullabc: clean
	rsync -av \
	--update \
	--exclude=.git \
	--exclude=*pth \
	--exclude=*idea \
	--exclude=*ignore \
	--exclude=*.ipynb \
	--exclude=*DS_Store \
	--exclude=__pycache__ \
	--exclude=*ipynb_checkpoints \
	acb11402ci@abc:/home/acb11402ci/checkpoints /home/quang/checkpoints/abci

yagi21: clean
	rsync -av \
	--exclude=*pyc \
	--exclude=*idea \
	--exclude=*ignore \
	--exclude=*.json \
	--exclude=*.ipynb \
	--exclude=*DS_Store \
	--exclude=__pycache__ \
	--exclude=*ipynb_checkpoints \
	$(CUR_DIR) administrator@yagi21:/home/administrator/quang/workspace/repos

pull22: clean
	rsync -av \
	--update \
	--exclude=.git \
	--exclude=*pyc \
	--exclude=*idea \
	--exclude=*ignore \
	--exclude=*.json \
	--exclude=*.ipynb \
	--exclude=*DS_Store \
	--exclude=__pycache__ \
	--exclude=*ipynb_checkpoints \
	administrator@yagi22:/home/administrator/quang/workspace/repos/visdial /home/quang/workspace/repos

clean:
	find '.' -name '*DS_Store' -exec rm -r {} +
	find '.' -name '*__pycache__' -exec rm -r {} +

git: clean
	git add .
	git commit -m "$m"
	git push -u origin master
	
all: clean git