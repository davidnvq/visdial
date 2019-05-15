import os
import argparse
from .colab import *
# TODO remove . in colab

parser = argparse.ArgumentParser()
parser.add_argument("--overfit", action="store_true")
parser.add_argument("--gdrive", action="store_true")
parser.add_argument("--split", default="val")
parser.add_argument("--encoder", default="lf")
parser.add_argument("--decoder", default="disc")
parser.add_argument("--ckpt-path", default="")
parser.add_argument("--load-pthpath", default="")
args = parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument("--config-yml", default="")
parser.add_argument("--json-dialogs", default="")
parser.add_argument("--json-dense", default="")
parser.add_argument("--load-pthpath", default="")
parser.add_argument("--save-ranks-path", default="logs/ranks.json")


def get_data_path_dict():
	dir_path = os.path.join(ROOT, 'datasets/visdial')

	data_path_dict = {
		'json-dialogs'     : os.path.join(dir_path, 'visdial_1.0_val.json'),
		'json-word-counts' : os.path.join(dir_path, 'visdial_1.0_word_counts_train.json'),
		'json-dense'       : os.path.join(dir_path, 'visdial_1.0_val_dense_annotations.json'),
		'image-features-h5': os.path.join(dir_path, 'features_faster_rcnn_x101_val.h5'),
		}
	if args.split == 'test':
		data_path_dict.update({
			'json-dense'       : '',
			'json-dialogs'     : os.path.join(dir_path, 'visdial_1.0_test.json'),
			'image-features-h5': os.path.join(dir_path, 'features_faster_rcnn_x101_test.h5')
			})

	return data_path_dict


def get_other_arg_dict():
	config_name = f'{args.encoder}_{args.decoder}_faster_rcnn_x101.yml'
	config_yml = os.path.join(PATH_PROJ, 'configs', config_name)

	other_arg_dict = {
		'gpu-ids'    : 0,
		'step-size'  : 2,
		'cpu-workers': 4,
		'validate'   : True,
		'config-yml' : config_yml,
		'overfit'    : args.overfit,
		}
	return other_arg_dict


def get_ckpt_path_dict(root_path):
	ckpt_dir = f'{args.encoder}_{args.decoder}'
	ckpt_path = os.path.join(root_path, 'checkpoints', ckpt_dir, args.ckpt_path)
	ckpt_path_dict = {
		'save-ranks-path': os.path.join(ckpt_path, f'{args.split}_ranks.json'),
		'load-pthpath'   : os.path.join(ckpt_path, args.load_pthpath)
		}
	return ckpt_path_dict


def get_args_dict():
	args_dict = {}
	data_path_dict = get_data_path_dict()
	ckpt_path_dict = get_ckpt_path_dict(GDRIVE if args.gdrive else ROOT)
	other_arg_dict = get_other_arg_dict()

	args_dict.update(data_path_dict)
	args_dict.update(ckpt_path_dict)
	args_dict.update(other_arg_dict)
	return args_dict


def evaluate(args_dict):
	evaluate_cmd = f'python {PATH_PROJ}/evaluate.py '
	print(evaluate_cmd)
	for arg in args_dict:
		value = args_dict[arg]
		if isinstance(value, bool):
			evaluate_cmd += f'--{arg} '
		else:
			evaluate_cmd += f'--{arg} '
			if isinstance(value, list):
				evaluate_cmd += f'{" ".join([str(v) for v in value])} '
			else:
				evaluate_cmd += f'{value} '
		print('--{:<25}: {}'.format(arg, value))
	execute_cmd(evaluate_cmd, verbose=False)


if __name__ == '__main__':
	# install_packages()
	# pull_project()
	# mount_gdrive()
	args_dict = get_args_dict()
	download_dataset()
	evaluate(args_dict)
