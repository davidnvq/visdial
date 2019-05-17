import os
import argparse
from colab import *

parser = argparse.ArgumentParser()
parser.add_argument("--overfit", action="store_true")
parser.add_argument("--gdrive", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--encoder", default="lf")
parser.add_argument("--decoder", default="disc")
parser.add_argument("--batch-size", default=16, type=int)
parser.add_argument("--num-epochs", default=10, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--lr-steps", default=[5, ], nargs='+', type=int)
parser.add_argument("--ckpt-path", default="")
parser.add_argument("--load-pthpath", default="")
args = parser.parse_args()

def get_data_path_dict():
	data_names = {
		'json-train'          : 'visdial_1.0_train.json',
		'json-val'            : 'visdial_1.0_val.json',
		'json-word-counts'    : 'visdial_1.0_word_counts_train.json',
		'json-val-dense'      : 'visdial_1.0_val_dense_annotations.json',
		'image-features-va-h5': 'features_faster_rcnn_x101_val.h5',
		'image-features-tr-h5': 'features_faster_rcnn_x101_train.h5',
		}
	data_path_dict = {k: os.path.join(ROOT, 'datasets/visdial', v)
	                  for k, v in data_names.items()}
	return data_path_dict


def get_other_arg_dict():
	config_name = f'{args.encoder}_{args.decoder}_faster_rcnn_x101.yml'
	config_yml = os.path.join(PATH_PROJ, 'configs', config_name)

	if args.overfit:
		comet_name = 'test'
	else:
		comet_name = f'{args.encoder}-{args.decoder}'

	other_arg_dict = {
		'gpu-ids'    : 0,
		'step-size'  : 2,
		'cpu-workers': 4,
		'validate'   : True,
		'comet-name' : comet_name,
		'config-yml' : config_yml,
		'overfit'    : args.overfit,
		'lr'         : 1e-3,
		'lr-steps'   : args.lr_steps,
		'num-epochs' : args.num_epochs,
		'batch-size' : args.batch_size,
		}
	return other_arg_dict


def get_ckpt_path_dict(root_path):
	ckpt_dir = f'{args.encoder}_{args.decoder}'
	ckpt_path = os.path.join(root_path, 'checkpoints', ckpt_dir, args.ckpt_path)
	ckpt_path_dict = {
		'save-dirpath': ckpt_path,
		'load-pthpath': os.path.join(ckpt_path, args.load_pthpath)
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


def train(args_dict):
	train_cmd = f'python {PATH_PROJ}/train.py '
	print('\n' + train_cmd)
	for arg in args_dict:
		value = args_dict[arg]
		if isinstance(value, bool):
			if value is True:
				train_cmd += f'--{arg} '
		else:
			train_cmd += f'--{arg} '
			if isinstance(value, list):
				train_cmd += f'{" ".join([str(v) for v in value])} '
			else:
				train_cmd += f'{value} '
		print('--{:<25}: {}'.format(arg, value))
	execute_cmd(train_cmd, verbose=False)


if __name__ == '__main__':
	mount_gdrive()
	install_packages()
	# pull_project()
	args_dict = get_args_dict()
	download_dataset(train=True)
	train(args_dict)
