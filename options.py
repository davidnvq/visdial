from comet_ml import Experiment, OfflineExperiment

import os
import yaml
import argparse
osp = os.path.join

HOME_PATH = '/media/local_workspace/quang'
DATA_PATH = f'{HOME_PATH}/datasets/visdial'
LOG_PATH = '/home/quang/workspace/log/tensorboard'

os.system(f"mkdir -p {HOME_PATH}")
os.system(f"mkdir -p {DATA_PATH}")
os.system(f"mkdir -p {LOG_PATH}")

def get_comet_experiment(config):
    print("GET COMET!")
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
    print("LOG PROJECT!")
    config_name = config['config_name'].replace('/', '_')
    os.system(f"zip -qr {config_name}.zip /home/quang/workspace/repos/visdial -x '*.git*' '*.ipynb_checkpoints*' '*.idea*' '*__pycache__*'")
    experiment.log_asset(f"{config_name}.zip")
    os.system(f"mv {config_name}.zip {config['callbacks']['log_dir']}")

    # Log parameters to comet
    for key in config:
        if not isinstance(config[key], dict):
            experiment.log_parameter(key, config[key])

    for key in ['model', 'solver']:
        experiment.log_parameters(config[key])

    return experiment

def get_training_config_and_args():
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

    print("LOG CONFIGURATION!")
    for dir in [config['callbacks']['log_dir'], config['callbacks']['save_dir']]:

        # os.system(f"rm -r {dir}")

        if not os.path.exists(dir):
            os.system(f"mkdir -p {dir}")

        config_name = args.config_name.replace('/', '_')
        with open(f'{dir}/{config_name}.yml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=4)


    print(yaml.dump(config, default_flow_style=False, indent=4))

    dir = '/home/quang/workspace/repos/visdial/configs'
    if not os.path.exists(dir):
        os.system(f'mkdir -p {dir}')

    with open(f'{dir}/{config_name}.yml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=4)

    return config, args


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
    """Label Smoothing Loss"""
    group.add_argument('--ls_epsilon', default=None, type=float, metavar='N')

    """Knowledge Distillation"""
    group.add_argument('--kd_alpha', default=None, type=float, metavar='N')
    group.add_argument('--kd_temperature', default=None, type=float, metavar='N')

    """Adam Optimizer"""
    group.add_argument('--optimizer', default='adam',
                       choices=['sgd', 'adam', 'adamax'])
    group.add_argument('--adam_betas', nargs='+', type=float, default=[0.9, 0.997])
    group.add_argument('--adam_eps', type=float, default=1e-9)
    group.add_argument('--weight_decay', '--wd', default=1e-5, type=float, metavar='WD',
                       help='weight decay')
    group.add_argument('--clip_norm', default=None, type=float,
                       metavar='N',
                       help='clip threshold of gradients')

    """Dataloader"""
    group.add_argument('--num_epochs', default=30, type=int, metavar='N',
                       help='Total number of epochs')
    group.add_argument('--batch_size', default=8, type=int,
                       metavar='N',
                       help="Batch_size for training")
    group.add_argument('--cpu_workers', default=8, type=int)
    group.add_argument('--batch_size_multiplier', default=1, type=int,
                       metavar='N',
                       help='Cumsum of loss in N batches and update optimizer once')

    """Learning Rate Scheduler"""
    group.add_argument('--scheduler_type', default='CosineLR',
                       help='learning rate scheduler type',
                       choices=['CosineLR', 'LinearLR'])
    group.add_argument('--init_lr', default=5e-3, type=float,
                       help='initial learning rate')
    group.add_argument('--min_lr', default=1e-5, type=float, metavar='LR',
                       help='minimum learning rate')
    group.add_argument('--num_samples', default=123287, type=int,
                       help='The number of training samples')

    """Warmup Scheduler"""
    group.add_argument('--warmup_factor', default=0.2, type=float,
                       metavar='N',
                       help='lr will increase from 0 -> init_lr with warm_factor:'
                            'after every batch, lr = lr * warmup_factor')
    group.add_argument('--warmup_epochs', default=1, type=int, metavar='N')

    """Linear Scheduler"""
    group.add_argument('--linear_gama', default=None, type=float, metavar='LG',
                       help='learning rate shrink factor for step reduce, lr_new = (lr * lr_gama) at milestone step')
    group.add_argument('--milestone_steps', nargs='+', type=int, metavar='LS', default=None,
                        help='If we use step_lr_scheduler rather than cosine')
    group.add_argument('--fp16', default=False, action='store_true')
    return group


def add_callback_args(parser):
    group = parser.add_argument_group('callbacks')
    group.add_argument('--resume', default=False, action='store_true')
    group.add_argument('--online', default=False, action='store_true')
    group.add_argument('--validate', default=True, action='store_true')
    group.add_argument('--path_pretrained_ckpt', metavar='DIR', default=None,
                       help='filename in save-dir from which to load checkpoint, checkpoint_last.pt')
    return group


def add_model_args(parser):
    group = parser.add_argument_group('model')

    group.add_argument('--decoder_type', choices=['misc', 'disc', 'gen'], default='misc', help='Type of decoder')
    group.add_argument('--hidden_size', type=int, metavar='N', default=512)
    group.add_argument('--dropout', type=float, metavar='N', default=0.1)
    group.add_argument('--test_mode', action='store_true', default=False)

    """Image Feature"""
    group.add_argument('--img_feat_size', type=int, metavar='N', default=2048)
    group.add_argument('--img_num_attns', type=int, metavar='N',default=None)
    group.add_argument('--img_has_bboxes', action='store_true', default=False)
    group.add_argument('--img_has_attributes', action='store_true', default=False)
    group.add_argument('--img_has_classes', action='store_true', default=False)

    """Text Feature"""
    group.add_argument('--txt_vocab_size', type=int, metavar='N',default=11322)
    group.add_argument('--txt_tokenizer', choices=['nlp', 'bert'], default='nlp')
    group.add_argument('--txt_bidirectional', action='store_true', default=True)
    group.add_argument('--txt_embedding_size', type=int, default=300)
    group.add_argument('--txt_has_pos_embedding', action='store_true', default=False)
    group.add_argument('--txt_has_layer_norm', action='store_true', default=False)
    group.add_argument('--txt_has_decoder_layer_norm', action='store_true', default=False)

    """Cross-Attention"""
    group.add_argument('--ca_has_shared_attns', action='store_true', default=False)
    group.add_argument('--ca_has_updated_hist', action='store_true', default=False)
    group.add_argument('--ca_has_proj_linear', action='store_true', default=False)
    group.add_argument('--ca_has_layer_norm', action='store_true', default=False)
    group.add_argument('--ca_has_residual', action='store_true', default=False)
    group.add_argument('--ca_num_cross_attns', type=int, metavar='N', default=1)
    group.add_argument('--ca_num_cross_attn_heads', type=int, metavar='N', default=4)
    group.add_argument('--ca_memory_size', type=int, default=2)
    return group