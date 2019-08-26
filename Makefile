.PHONY: clean, k2, clean, all

k2: clean
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
	../visdial quang@k2:/home/quang/repos

clean:
	find '.' -name '*DS_Store' -exec rm -r {} +
	find '.' -name '*__pycache__' -exec rm -r {} +

git: clean
	git add .
	git commit -m "$m"
	git push -u origin master
	
all: k2 git