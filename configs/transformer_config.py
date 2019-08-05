import json
import copy
import os

osp = os.path.join

HOME_PATH = '/home/quanguet'
DATA_PATH = osp(HOME_PATH, 'datasets/visdial')
print('HOME_PATH', HOME_PATH)
print('DATA_PATH', DATA_PATH)

HIDDEN_SIZE = 512
EMBEDDING_SIZE = 300
VOCAB_SIZE = 11322
DROPOUT = 0.2
NUM_LSTM_LAYERS = 2

NUM_SELF_HEADS = 8
NUM_SELF_ATTNS = 4

NUM_CROSS_HEADS = 4
NUM_CROSS_ATTNS = 2
MEMORY_SIZE = 2

CONFIG = {
	'seed'     : 0,

	'callbacks': {
		'validate'            : True,
		'resume'              : False,
		'comet_project'       : 'lf-bert-disc',
		'path_pretrained_ckpt': '',
		'path_dir_save_ckpt'  : osp(HOME_PATH, 'checkpoints/visdial/lf_disc/lf_bert_disc')
		},

	'dataset'  : {

		'overfit'          : True,
		'img_norm'         : 1,
		'concat_history'   : True,
		'batch_size'       : 2,
		'cpu_workers'      : 4,
		'max_seq_len'      : 25,
		'is_return_options': True,
		'is_add_boundaries': True,

		'train'            : {
			'path_feat_img'          : osp(DATA_PATH, 'features_faster_rcnn_x101_val.h5'),
			'path_json_dialogs'      : osp(DATA_PATH, 'visdial_1.0_val.json'),
			'path_feat_history'      : osp(DATA_PATH, 'features_bert_train_history.h5'),
			'path_feat_answers'      : osp(DATA_PATH, 'features_bert_train_answers.h5'),
			'path_feat_questions'    : osp(DATA_PATH, 'features_bert_train_questions.h5'),
			'path_json_dense_dialogs': osp(DATA_PATH, ''),
			'path_json_word_count'   : osp(DATA_PATH, 'visdial_1.0_word_counts_train.json')

			},
		'val'              : {
			'path_feat_img'          : osp(DATA_PATH, 'features_faster_rcnn_x101_val.h5'),
			'path_json_dialogs'      : osp(DATA_PATH, 'visdial_1.0_val.json'),
			'path_feat_history'      : osp(DATA_PATH, ''),
			'path_feat_answers'      : osp(DATA_PATH, ''),
			'path_feat_questions'    : osp(DATA_PATH, ''),
			'path_json_dense_dialogs': osp(DATA_PATH, 'visdial_1.0_val_dense_annotations.json')
			},

		'test'             : {
			'path_feat_img'          : osp(DATA_PATH, 'features_faster_rcnn_x101_test.h5'),
			'path_feat_history'      : osp(DATA_PATH, 'features_bert_test_dialogs.h5'),
			'path_feat_answers'      : osp(DATA_PATH, 'features_bert_test_answers.h5'),
			'path_feat_questions'    : osp(DATA_PATH, 'features_bert_test_questions.h5'),
			'path_json_dense_dialogs': osp(DATA_PATH, '')

			}
		},

	'model'    : {
		'tokenizer'      : 'nlp',
		'hidden_size'    : HIDDEN_SIZE,
		'dropout'        : DROPOUT,
		'vocab_size'     : VOCAB_SIZE,  # 11322 for nlp

		'text_embeddings': {
			'vocab_size'      : VOCAB_SIZE,
			'embedding_size'  : EMBEDDING_SIZE,
			'hidden_size'     : HIDDEN_SIZE,
			'has_position'    : True,
			'has_hidden_layer': True,
			},

		'get_transformer': {
			'hidden_size'   : HIDDEN_SIZE,
			'num_heads'     : NUM_SELF_HEADS,
			'd_ff'          : HIDDEN_SIZE,
			'dropout'       : DROPOUT,
			'num_self_attns': NUM_SELF_ATTNS,
			},

		'encoder'        : {
			'attn_encoder': {
				'hidden_size'    : HIDDEN_SIZE,
				'memory_size'    : MEMORY_SIZE,
				'num_heads'      : NUM_CROSS_ATTNS,
				'dropout'        : DROPOUT,
				'num_cross_attns': NUM_CROSS_ATTNS
				},

			'img_encoder' : {
				'img_feat_size': 2048,
				'hidden_size'  : HIDDEN_SIZE,
				'dropout'      : DROPOUT
				},

			'text_encoder': {

				'hist_encoder': {
					'type': 'transformer',
					},

				'ques_encoder': {
					'type': 'transformer',
					},
				}
			},

		'decoder'        : {
			'disc': {
				'opt_encoder': {
					'type': 'transformer'
					}
				},

			'gen' : {
				'dropout'        : DROPOUT,
				'vocab_size'     : VOCAB_SIZE,
				'hidden_size'    : HIDDEN_SIZE,
				'num_lstm_layers': NUM_LSTM_LAYERS,
				}
			}
		},

	'solver'   : {
		'device'         : 'cuda',
		'gpu_ids'        : [0, 1],
		'num_epochs'     : 20,
		'init_lr'        : 5e-4,
		'lr_steps'       : [15],
		'training_splits': 'train'
		}
	}


def save_config(config, config_path):
	pass


def get_config(config_path=None, is_save_config=False):
	if config_path is None:
		config = copy.deepcopy(CONFIG)

	if is_save_config:
		save_config(config, config_path)

	return config
