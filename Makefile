.PHONY: clean, yagi22, pull22, clean, all

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
	../visdial administrator@yagi22:/home/administrator/quang/workspace/repos



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