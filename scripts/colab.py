import os

# TODO: Change ROOT, uncomment os.system(cmd)
# TODO: Change PATH_PROJ

ROOT = '/content'
GDRIVE = os.path.join(ROOT, 'gdrive/My\ Drive')
PATH_PROJ = os.path.join(ROOT, 'visdial')


def execute_cmd(cmd, verbose=True):
	if verbose:
		print(cmd)
	os.system(cmd)

def pull_project():
	execute_cmd('cd /content/visdial; git pull https://github.com/quanguet/visdial.git')


def install_packages():
	execute_cmd('pip install comet-ml --quiet')
	execute_cmd('pip install tensorboardX --quiet')


def mount_gdrive():
	try:
		from google.colab import drive

		drive.mount('/content/gdrive')

	except ModuleNotFoundError:
		raise ModuleNotFoundError


def file_exists(file_path):
	if os.path.exists(file_path):
		print('{: <50}: exists!'.format(file_path))
	return os.path.exists(file_path)


def download_dataset(train=False):
	dir_path = os.path.join(ROOT, 'datasets', 'visdial')
	file_paths = [
		os.path.join(dir_path, 'features_faster_rcnn_x101_val.h5'),
		os.path.join(dir_path, 'features_faster_rcnn_x101_test.h5'),
		os.path.join(dir_path, 'visdial_1.0_word_counts_train.json'),
		os.path.join(dir_path, 'visdial_1.0_val_dense_annotations.json'),
		os.path.join(dir_path, 'visdial_1.0_val.json'),
		os.path.join(dir_path, 'visdial_1.0_train.json'),
		os.path.join(dir_path, 'features_faster_rcnn_x101_train.h5'),
		]

	file_links = [
		'https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_val.h5',
		'https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_test.h5',
		'https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/visdial_1.0_word_counts_train.json',
		'https://www.dropbox.com/s/3knyk09ko4xekmc/visdial_1.0_val_dense_annotations.json?dl=0',
		'https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=0',
		'https://www.dropbox.com/s/ix8keeudqrd8hn8/visdial_1.0_train.zip?dl=0',
		'https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_train.h5',
		]

	if train is False:
		file_paths == file_paths[:-2]

	for i in range(len(file_paths)):

		if not file_exists(file_paths[i]):
			print('Downloading {} ...'.format(file_paths[i]))
			if not file_exists(os.path.dirname(file_paths[i])):
				os.makedirs(os.path.dirname(file_paths[i]))

			if 'zip' in file_links[i]:
				file_paths[i] = file_paths[i].replace('json', 'zip')
			execute_cmd('wget {} --output-document={}'.format(file_links[i], file_paths[i]))
			print('Finished downloading!\n')

	for file in file_paths:
		if 'zip' in file:
			execute_cmd('unzip {}'.format(file))
			execute_cmd('rm {}'.format(file))

