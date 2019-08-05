import json
import copy
import os

osp = os.path.join

HOME_PATH = os.path.expanduser("~")
HOME_PATH = '/media/local_workspace/quang'
DATA_PATH = osp(HOME_PATH, 'datasets/visdial')


HIDDEN_SIZE = 512
EMBEDDING_SIZE = 300
VOCAB_SIZE = 11322
DROPOUT = 0.2
NUM_LSTM_LAYERS = 2

CONFIG = {
	'seed'         : 0,
	'comet_project': 'tmp',
	'config_name'  : 'lf_misc_lstm',

	'callbacks'    : {
		'validate'            : True,
		'resume'              : True,
		'path_pretrained_ckpt': osp(HOME_PATH, 'checkpoints/visdial/lf_misc_lstm/checkpoint_last.pth'),
		'path_dir_save_ckpt'  : osp(HOME_PATH, 'checkpoints/visdial/lf_misc_lstm')
		},

	'model'        : {
		'tokenizer'       : 'nlp',
		'hidden_size'     : 512,
		'dropout'         : 0.2,
		'vocab_size'      : 11322,
		'img_feature_size': 2048,
		'embedding_size'  : 300,
		'bidirectional'   : True,
		'encoder_type'    : 'lf',
		'decoder_type'    : 'misc'
		},

	'solver'       : {
		'device'         : 'cuda',
		'num_epochs'     : 25,
		'batch_size'     : 8,
		'init_lr'        : 1e-3,
		'lr_steps'       : [10, 15],
		'training_splits': 'train'
		},

	'dataset'      : {

		'overfit'          : False,
		'img_norm'         : 1,
		'concat_history'   : True,
		'cpu_workers'      : 8,
		'max_seq_len'      : 25,
		'is_return_options': True,
		'is_add_boundaries': True,

		'glove_path'       : osp(HOME_PATH, 'datasets/glove/glove840B_11322words.pkl'),

		'train'            : {
			'path_feat_img'          : osp(DATA_PATH, 'features_faster_rcnn_x101_train.h5'),
			'path_json_dialogs'      : osp(DATA_PATH, 'visdial_1.0_train.json'),
			'path_json_word_count'   : osp(DATA_PATH, 'visdial_1.0_word_counts_train.json'),
			'path_json_dense_dialogs': osp(DATA_PATH, '')

			},
		'val'              : {
			'path_feat_img'          : osp(DATA_PATH, 'features_faster_rcnn_x101_val.h5'),
			'path_json_dialogs'      : osp(DATA_PATH, 'visdial_1.0_val.json'),
			'path_json_dense_dialogs': osp(DATA_PATH, 'visdial_1.0_val_dense_annotations.json')
			},

		'test'             : {
			'path_feat_img'          : osp(DATA_PATH, 'features_faster_rcnn_x101_test.h5'),
			'path_json_dialogs'      : osp(DATA_PATH, 'visdial_1.0_test.json'),
			'path_json_dense_dialogs': osp(DATA_PATH, '')
			}
		},

	}


def save_config(config, config_path):
	pass


def get_lf_misc_lstm_config(config_path=None, is_save_config=False):
	if config_path is None:
		config = copy.deepcopy(CONFIG)

	if is_save_config:
		save_config(config, config_path)

	return config
