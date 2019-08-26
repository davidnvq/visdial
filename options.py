from comet_ml import Experiment, OfflineExperiment

import os
import yaml
import argparse
osp = os.path.join

HOME_PATH = '/media/local_workspace/quang'
DATA_PATH = f'{HOME_PATH}/datasets/visdial'
LOG_PATH = '/home/quang/workspace/log/tensorboard'


def get_comet_experiment(config):
    if config['callbacks']['online']:
        experiment = Experiment(api_key='2z9VHjswAJWF1TV6x4WcFMVss',
                                project_name='temp',
                                workspace='lightcv')
    else:
        experiment = OfflineExperiment(
            project_name='temp',
            workspace='lightcv',
            offline_directory=config['callbacks']['log_dir'])

    # Log project to comet
    os.system(f"zip -qr {config['config_name']}.zip ./ -x '*.git*' '*.ipynb_checkpoints*' '*.idea*' '*__pycache__*'")
    experiment.log_asset(f"{config['config_name']}.zip")
    os.system(f"mv {config['config_name']}.zip {config['callbacks']['log_dir']}")

    # Log parameters to comet
    for key in config:
        if not isinstance(config[key], dict):
            experiment.log_parameter(key, config[key])

    for key in ['model', 'solver']:
        experiment.log_parameters(config[key])

    return experiment

def get_training_config():
    parser = get_training_parser()

    args = parser.parse_args()
    # print([group.title for group in parser._action_groups])

    config = {}
    for group in parser._action_groups:
        if group.title in ['optional arguments', 'positional arguments']:
            group_dict = {arg.dest: getattr(args, arg.dest, None) for arg in group._group_actions}
            for key in group_dict:
                config[key] = group_dict[key]
                print(key)
        else:
            group_dict = {arg.dest:getattr(args, arg.dest, None) for arg in group._group_actions}
        config[group.title] = group_dict

    config['callbacks']['log_dir'] = osp(LOG_PATH, args.config_name)
    config['callbacks']['save_dir'] = osp(HOME_PATH, 'checkpoints/visdial/CVPR', args.config_name)

    for dir in [config['callbacks']['log_dir'], config['callbacks']['save_dir']]:
        if not os.path.exists(dir):
            os.system(f"mkdir -p {dir}")
        with open(f'{dir}/{args.config_name}.yml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=4)

    print(yaml.dump(config, default_flow_style=False, indent=4))

    # Copy project
    os.system(f"rsync -av \
    	--exclude=.git \
    	--exclude=*pyc \
    	--exclude=*idea \
    	--exclude=*ignore \
    	--exclude=*.ipynb \
    	--exclude=*DS_Store \
    	--exclude=__pycache__ \
    	--exclude=*ipynb_checkpoints \
    	./ {config['callbacks']['log_dir']}/source_code")

    return config


def get_training_parser():
    parser = get_parser('Trainer')
    add_dataset_args(parser)
    add_model_args(parser)
    add_solver_args(parser)
    add_callback_args(parser)
    parser.add_argument('--config_name', metavar='S', default='c1.0.0')
    return parser


def get_parser(desc):
    parser = argparse.ArgumentParser(description='Visual Dialog Toolkit -- ' + desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', metavar='N', type=int,
                        default=0)
    return parser


def add_args(parser):
    group = parser.add_argument_group("")
    group.add_argument('--', help='')
    return group


def add_dataset_args(parser):
    group = parser.add_argument_group("dataset")
    group.add_argument('--overfit', action='store_true',
                       default=False,
                       help='overfit on small dataset')
    group.add_argument('--concat_hist', action='store_true',
                       default=False,
                       help='concat history rounds into a single vector')
    group.add_argument('--max_seq_len', type=int, metavar='N',
                       default=20,
                       help='max number of tokens in a sequence')
    group.add_argument('--vocab_min_count', type=int, metavar='N',
                       default=5,
                       help='The word with frequency of 5 times will be listed in Vocabulary')
    group.add_argument('--is_legacy', default=False, action='store_true')
    group.add_argument('--img_norm', default=True, action='store_true')
    group.add_argument('--finetune', default=False, action='store_true')
    group.add_argument('--is_add_boundaries', default=True, action='store_true')
    group.add_argument('--is_return_options', default=True, action='store_true')
    group.add_argument('--num_boxes', choices=['fixed', 'adaptive'],
                       default='fixed', metavar='S',
                       help='The number of boxes per image from Faster R-CNN')
    group.add_argument('--glove_path', metavar='PATH',
                       default=osp(DATA_PATH, 'glove/embedding_Glove_840_300d.pkl'))
    group.add_argument('--train_feat_img_path', metavar='PATH',
                       default=osp(DATA_PATH, 'legacy/features_faster_rcnn_x101_train.h5'))
    group.add_argument('--val_feat_img_path', metavar='PATH',
                       default=osp(DATA_PATH, 'legacy/features_faster_rcnn_x101_val.h5'))
    group.add_argument('--train_json_dialog_path', metavar='PATH',
                       default=osp(DATA_PATH, 'annotations/visdial_1.0_train.json'))
    group.add_argument('--val_json_dialog_path', metavar='PATH',
                       default=osp(DATA_PATH, 'annotations/visdial_1.0_val.json'))
    group.add_argument('--val_json_dense_dialog_path', metavar='PATH',
                       default=osp(DATA_PATH, 'annotations/visdial_1.0_val_dense_annotations.json'))
    group.add_argument('--train_json_word_count_path', metavar='PATH',
                       default=osp(DATA_PATH, 'annotations/visdial_1.0_word_counts_train.json'))
    return group


def add_solver_args(parser):
    group = parser.add_argument_group('solver')
    group.add_argument('--lr', default=0.001, type=float,
                       help='initial learning rate')
    group.add_argument('--num_epochs', default=30, type=int, metavar='N',
                       help='Total number of epochs')
    group.add_argument('--batch_size', default=32, type=int,
                       metavar='N',
                       help="Batch_size for training")
    group.add_argument('--cpu_workers', default=8, type=int)
    group.add_argument('--batch_size_multiplier', default=1, type=int,
                       metavar='N',
                       help='Cumsum of loss in N batches and update optimizer once')
    group.add_argument('--optimizer', default='adamax',
                       choices=['sgd', 'adam', 'adamax'])
    group.add_argument('--clip_norm', default=25.0, type=float,
                       metavar='N',
                       help='clip threshold of gradients')
    group.add_argument('--momentum', default=0.99, type=float, metavar='M',
                       help='momentum factor')
    group.add_argument('--weight_decay', '--wd', default=0.0, type=float, metavar='WD',
                       help='weight decay')
    group.add_argument('--lr_scheduler', default='consine_warmup',
                       help='learning rate scheduler type',
                       choices=['consine_warmup', 'step'])
    group.add_argument('--min_lr', default=1e-5, type=float, metavar='LR',
                       help='minimum learning rate')
    group.add_argument('--warmup_factor', default=0.2, type=float,
                       metavar='N',
                       help='lr will increase from 0 -> init_lr with warm_factor:'
                            'after every batch, lr = lr * warmup_factor')
    group.add_argument('--warmup_epochs', default=1, type=int, metavar='N')
    group.add_argument('--lr_gama', default=0.5, type=float, metavar='LG',
                       help='learning rate shrink factor for step reduce, lr_new = (lr * lr_gama) at milestone step')
    group.add_argument('--lr_steps', nargs='+', type=int, metavar='LS',
                        default=[4, 8, 12, 16, 20, 24, 26],
                        help='If we use step_lr_scheduler rather than cosine')
    return group


def add_callback_args(parser):
    group = parser.add_argument_group('callbacks')
    group.add_argument('--resume', default=False, action='store_true')
    group.add_argument('--online', default=False, action='store_true')
    group.add_argument('--validate', default=True, action='store_true')
    group.add_argument('--path_pretrained_ckpt', metavar='DIR', default=None,
                       help='filename in save-dir from which to load checkpoint, checkpoint_last.pt')
    return group


def add_interactive_args(parser):
    group = parser.add_argument_group('Interactive')
    group.add_argument('--buffer-size', default=0, type=int, metavar='N',
                       help='read this many sentences into a buffer before processing them')


def add_model_args(parser):
    group = parser.add_argument_group('model')

    group.add_argument('--decoder_type', choices=['misc', 'disc', 'gen'],
                       default='misc',
                       help='Type of decoder')
    group.add_argument('--hidden_size', type=int,
                       metavar='N',
                       default=512,
                       help='Hidden size in all layers')
    group.add_argument('--img_feature_size', type=int,
                       metavar='N',
                       default=2048,
                       help='The image feature size from Faster R-CNN(2048)')
    group.add_argument('--vocab_size', type=int,
                       metavar='N',
                       default=11322,
                       help='The vocabulary size, input_size for embedding')
    group.add_argument('--dropout', type=float,
                       metavar='N',
                       default=0.2)
    group.add_argument('--tokenizer', choices=['nlp', 'bert'],
                       default='nlp')
    group.add_argument('--num_cross_attns', type=int,
                       metavar='N',
                       default=1)
    group.add_argument('--num_cross_attn_heads', type=int, metavar='N', default=4)
    group.add_argument('--share_attn_weights', action='store_true',
                       default=False)
    group.add_argument('--has_residual', action='store_true', default=False)
    group.add_argument('--embedding_size', type=int,
                       default=300)
    group.add_argument('--memory_size', type=int,
                       default=2)
    group.add_argument('--embedding_has_position', action='store_true', default=True)
    group.add_argument('--embedding_has_hidden_layer', action='store_true', default=False)
    group.add_argument('--test_mode', action='store_true', default=False)
    group.add_argument('--bidirectional', action='store_true', default=True)


    return group

# get_training_config()