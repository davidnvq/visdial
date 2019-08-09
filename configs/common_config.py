import copy
import os

osp = os.path.join

HOME_PATH = '/media/local_workspace/quang'
HOME_PATH = '/home/quanguet'
DATA_PATH = f'{HOME_PATH}/datasets/visdial'

EXTENTION = 'v1.0'
CONFIG_NAME = 'attn_misc_lstm'

CONFIG = {
	'seed'         : 0,
	'config_name'  : CONFIG_NAME,
	'comet_project': 'tmp',

	'callbacks'    : {
		'validate'            : True,
		'resume'              : False,
		'path_pretrained_ckpt': '',
		'path_dir_save_ckpt'  : f'{HOME_PATH}/checkpoints/visdial/reborn/{CONFIG_NAME}_{EXTENTION}'
		},

	'solver'       : {
		'num_epochs' : 30,
		'batch_size' : 8,
		'cpu_workers': 16,
		'init_lr'    : 1e-3,
		'lr_steps'   : [4, 8, 12, 16, 20, 24, 26],
		},

	'model'        : {
		'tokenizer'                 : 'nlp',
		'hidden_size'               : 512,
		'dropout'                   : 0.2,
		'img_feature_size'          : 2048,
		'vocab_size'                : 11322,
		'bidirectional'             : True,
		'decoder_type'              : 'misc',
		'encoder_type'              : 'attn',
		'encoder_memory_size'       : 2,
		'encoder_num_heads'         : 4,
		'encoder_num_cross_attns'   : 2,
		'embedding_has_position'    : True,
		'embedding_has_hidden_layer': False,
		'embedding_size'            : 300,
		'share_attn'                : False,
		},

	'dataset'      : {
		'overfit'          : False,
		'img_norm'         : 1,
		'is_legacy'        : True,
		'concat_history'   : True,
		'max_seq_len'      : 20,
		'is_return_boxes'  : False,
		'num_boxes'        : 36,
		'is_return_options': True,
		'is_add_boundaries': True,
		'glove'            : osp(DATA_PATH, 'glove/embedding_Glove_840_300d.pkl'),
		'train'            : {
			'path_feat_img'          : osp(DATA_PATH, 'legacy/features_faster_rcnn_x101_train.h5'),
			'path_json_dialogs'      : osp(DATA_PATH, 'annotations/visdial_1.0_train.json'),
			'path_json_word_count'   : osp(DATA_PATH, 'annotations/visdial_1.0_word_counts_train.json'),
			'path_json_dense_dialogs': ''
			},

		'val'              : {
			'path_feat_img'          : osp(DATA_PATH, 'legacy/features_faster_rcnn_x101_val.h5'),
			'path_json_dialogs'      : osp(DATA_PATH, 'annotations/visdial_1.0_val.json'),
			'path_json_dense_dialogs': osp(DATA_PATH, 'annotations/visdial_1.0_val_dense_annotations.json')
			},

		'test'             : {
			'path_feat_img'          : osp(DATA_PATH, 'legacy/features_faster_rcnn_x101_test.h5'),
			'path_json_dialogs'      : osp(DATA_PATH, 'annotations/visdial_1.0_test.json'),
			'path_json_dense_dialogs': ''
			}
		}
	}


def get_attn_misc_lstm_config(config_path=None):
	if config_path is None:
		config = copy.deepcopy(CONFIG)

	return config
