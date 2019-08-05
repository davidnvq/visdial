from comet_ml import Experiment, OfflineExperiment
from tensorboardX import SummaryWriter

import os
import sys
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from visdial.model import get_model
from visdial.data.dataset import VisDialDataset
from visdial.metrics import SparseGTMetrics, NDCG
from visdial.utils.checkpointing import CheckpointManager, load_checkpoint_from_config
from visdial.utils import move_to_cuda

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def get_comet_experiment(config, is_online=False, offline_dir=None):
    if is_online:
        return Experiment(api_key='2z9VHjswAJWF1TV6x4WcFMVss',
                          project_name=config['comet_project'],
                          workspace='lightcv')
    else:
        return OfflineExperiment(
            project_name=config['comet_project'],
            workspace='lightcv',
            offline_directory=offline_dir)


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="attn_disc_lstm")
parser.add_argument('--online', action='store_true')
args = parser.parse_args()

# =============================================================================
#   INPUT ARGUMENTS AND CONFIG AND PARAMS
# =============================================================================

from configs import get_config

config = get_config(config_name=args.config)

save_dirpath = config['callbacks']['path_dir_save_ckpt']
log_path = save_dirpath + '/log'
logging.info(f"log_path: {log_path}")


os.makedirs(log_path, exist_ok=True)

experiment = get_comet_experiment(config, args.online, log_path)

for key in config:
    if not isinstance(config[key], dict):
        experiment.log_parameter(key, config[key])


for key in ['model', 'solver']:
    experiment.log_parameters(config[key])

# =============================================================================
# ZIPPING PROJECTS
# =============================================================================
logging.info("Start zipping project...")
os.system(f"zip -r -uq  {args.config}.zip /home/quang/repos/visdial -x '*.git*' '*.ipynb_checkpoints*' '*.idea*' '*__pycache__*' ")
experiment.log_asset(f"{args.config}.zip")
os.system(f"rm {args.config}.zip")
logging.info("Done zipping!")


# =============================================================================
# For reproducibility.
# =============================================================================
seed = config['seed']

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(seed)

# =============================================================================
#   SETUP DATASET, DATALOADER, MODEL, CRITERION, OPTIMIZER, SCHEDULER
# =============================================================================

logging.info(f"CUDA number: {torch.cuda.device_count()}")

"""DATASET INIT"""
logging.info("Loading train dataset...")
train_dataset = VisDialDataset(config, split='train')

train_dataloader = DataLoader(train_dataset,
                              batch_size=config['solver']['batch_size'] * torch.cuda.device_count(),
                              num_workers=config['dataset']['cpu_workers'],
                              shuffle=True)

logging.info("Loading val dataset...")
val_dataset = VisDialDataset(config, split='val')

val_dataloader = DataLoader(val_dataset,
                            batch_size=2 * torch.cuda.device_count(),
                            num_workers=config['dataset']['cpu_workers'])


"""MODEL INIT"""
logging.info("Init model...")
device = torch.device('cuda')
model = get_model(config)
logging.info("Move model to GPU...")
model = model.to(device)

"""LOSS FUNCTION"""
disc_criterion = nn.CrossEntropyLoss()
gen_criterion = nn.CrossEntropyLoss(ignore_index=0)

"""OPTIMIZER"""
optimizer = optim.Adam(model.parameters(), lr=config['solver']['init_lr'])
scheduler = MultiStepLR(optimizer, milestones=config['solver']['lr_steps'], gamma=0.5)

# =============================================================================
#   SETUP BEFORE TRAINING LOOP
# =============================================================================
summary_writer = SummaryWriter(log_dir=log_path)
checkpoint_manager = CheckpointManager(model, optimizer, save_dirpath, config=config)
sparse_metrics = SparseGTMetrics()
disc_metrics = SparseGTMetrics()
gen_metrics = SparseGTMetrics()
ndcg = NDCG()
disc_ndcg = NDCG()
gen_ndcg = NDCG()

logging.info("Loading checkpoints...")
start_epoch, model, optimizer = load_checkpoint_from_config(model, optimizer, config)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# =============================================================================
#   TRAINING LOOP
# =============================================================================

iterations = len(train_dataset) // (config['solver']['batch_size'] * torch.cuda.device_count()) + 1
num_examples = torch.tensor(len(train_dataset), dtype=torch.float)
global_iteration_step = start_epoch * iterations

for epoch in range(start_epoch, config['solver']['num_epochs']):
    logging.info(f"Training for epoch {epoch}:")


    with tqdm(total=iterations) as pbar:

        epoch_loss = torch.tensor(0.0)
        for i, batch in enumerate(train_dataloader):
            batch = move_to_cuda(batch, device)

            # zero out gradients
            optimizer.zero_grad()

            # do forward
            out = model(batch)

            # compute loss
            gen_loss = torch.tensor(0.0, requires_grad=True, device='cuda')
            disc_loss = torch.tensor(0.0, requires_grad=True, device='cuda')
            batch_loss = torch.tensor(0.0, requires_grad=True, device='cuda')
            if out.get('opt_scores') is not None:
                scores = out['opt_scores'].view(-1, 100)
                target = batch['ans_ind'].view(-1)

                sparse_metrics.observe(out['opt_scores'], batch['ans_ind'])
                disc_loss = disc_criterion(scores, target)
                batch_loss = batch_loss + disc_loss

            if out.get('ans_out_scores') is not None:
                scores = out['ans_out_scores'].view(-1, config['model']['vocab_size'])
                target = batch['ans_out'].view(-1)
                gen_loss = gen_criterion(scores, target)
                batch_loss = batch_loss + gen_loss

            # compute gradients
            batch_loss.backward()

            # update params
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix(epoch=epoch,
                             gen_loss=gen_loss.item(),
                             disc_loss=disc_loss.item(),
                             batch_loss=batch_loss.item())

            # log metrics
            experiment.log_metric('train/batch_loss', batch_loss.item())
            summary_writer.add_scalar(f'{args.config}-train/batch_loss', batch_loss.item(), global_iteration_step)


            global_iteration_step += 1
            torch.cuda.empty_cache()

            epoch_loss += batch["ans"].size(0) * batch_loss.detach()

    if out.get('opt_scores') is not None:
        avg_metric_dict = {}
        avg_metric_dict.update(sparse_metrics.retrieve(reset=True))

        for metric_name, metric_value in avg_metric_dict.items():
            logging.info(f"{metric_name}: {metric_value}")
            experiment.log_metric(f"train/{metric_name}", metric_value)

        summary_writer.add_scalars(f"{args.config}-train/metrics", avg_metric_dict, global_iteration_step)

    epoch_loss /= num_examples
    experiment.log_metric('train/epoch_loss', epoch_loss.item())
    logging.info(f"train/epoch_loss: {epoch_loss.item()}\n")
    summary_writer.add_scalar(f'{args.config}-train/epoch_loss', epoch_loss.item(), global_iteration_step)

    # log the parameters with histogram
    selected_names = ['module.encoder.attn_encoder.cross_attn_encoder.0.attns.0.x_memory',
                      'module.encoder.attn_encoder.cross_attn_encoder.1.attns.0.x_memory',
                      'module.encoder.attn_encoder.cross_attn_encoder.0.attns.0.x_proj_linear.weight',
                      'module.encoder.attn_encoder.cross_attn_encoder.1.attns.0.x_proj_linear.weight',
                      'module.encoder.attn_encoder.cross_attn_encoder.0.norms.0.linear.0.weight',
                      'module.encoder.attn_encoder.cross_attn_encoder.1.norms.0.linear.0.weight',
                      'module.encoder.summaries.0.attn_linear.0.weight',
                      'module.encoder.encoder_linear.0.weight',
                      'module.encoder.encoder_linear.2.weight'
                      ]

    for name, param in model.named_parameters():
        if name in selected_names:
            summary_writer.add_histogram(name, param.clone().cpu().data.numpy(), global_iteration_step)

    scheduler.step(epoch)

    # -------------------------------------------------------------------------
    #   ON EPOCH END  (checkpointing and validation)
    # -------------------------------------------------------------------------
    # Validate and report automatic metrics.

    if config['callbacks']['validate']:
        # Switch dropout, batchnorm etc to the correct mode.
        model.eval()

        logging.info(f"\nValidation after epoch {epoch}:")

        for batch in val_dataloader:
            move_to_cuda(batch, device)

            with torch.no_grad():
                out = model(batch)

                if out.get('opt_scores') is not None:
                    scores = out['opt_scores']
                    disc_metrics.observe(scores, batch["ans_ind"])

                    if "gt_relevance" in batch:
                        scores = scores[
                                 torch.arange(scores.size(0)),
                                 batch["round_id"] - 1, :]

                        disc_ndcg.observe(scores, batch["gt_relevance"])

                if out.get('opts_out_scores') is not None:
                    scores = out['opts_out_scores']
                    gen_metrics.observe(scores, batch["ans_ind"])

                    if "gt_relevance" in batch:
                        scores = scores[
                                 torch.arange(scores.size(0)),
                                 batch["round_id"] - 1, :]

                        gen_ndcg.observe(scores, batch["gt_relevance"])

                if out.get('opt_scores') is not None and out.get('opts_out_scores') is not None:
                    scores = (out['opts_out_scores'] + out['opt_scores']) / 2

                    sparse_metrics.observe(scores, batch["ans_ind"])
                    if "gt_relevance" in batch:
                        scores = scores[
                                 torch.arange(scores.size(0)),
                                 batch["round_id"] - 1, :]

                        ndcg.observe(scores, batch["gt_relevance"])

        avg_metric_dict = {}
        avg_metric_dict.update(sparse_metrics.retrieve(reset=True, key='avg_'))
        avg_metric_dict.update(ndcg.retrieve(reset=True, key='avg_'))

        disc_metric_dict = {}
        disc_metric_dict.update(disc_metrics.retrieve(reset=True, key='disc_'))
        disc_metric_dict.update(disc_ndcg.retrieve(reset=True, key='disc_'))

        gen_metric_dict = {}
        gen_metric_dict.update(gen_metrics.retrieve(reset=True, key='gen_'))
        gen_metric_dict.update(gen_ndcg.retrieve(reset=True, key='gen_'))

        for metric_dict in [avg_metric_dict, disc_metric_dict, gen_metric_dict]:
            for metric_name, metric_value in metric_dict.items():
                logging.info(f"{metric_name}: {metric_value}")
                experiment.log_metric(f"val/{metric_name}", metric_value)

            summary_writer.add_scalars(f"{args.config}-val/metrics", metric_dict, global_iteration_step)

        model.train()
        torch.cuda.empty_cache()

        # Checkpoint
        if 'disc' in config['config_name']:
            checkpoint_manager.step(epoch=epoch, only_best=False, metrics=disc_metric_dict, key='disc_')
        elif 'gen' in config['config_name']:
            checkpoint_manager.step(epoch=epoch, only_best=False, metrics=gen_metric_dict, key='gen_')

        elif 'misc' in config['config_name']:
            checkpoint_manager.step(epoch=epoch, only_best=False, metrics=disc_metric_dict, key='disc_')

    if args.online is False:
        logging.info('Copy log to local_computer...\n\n')

        folder_name = config['callbacks']['path_dir_save_ckpt'].split('/')[-1]
        os.system(f'scp -i /home/quang/.ssh/key_local_computer -r {log_path} quanguet@172.16.12.170:/home/quanguet/Dropbox/checkpoints/visdial/{folder_name}')

        log_files = [file for file in glob('/home/quang/log/*') if folder_name in file]
        for log_file in log_files:
            os.system(f'scp -i /home/quang/.ssh/key_local_computer -r {log_file} \
            quanguet@172.16.12.170:/home/quanguet/Dropbox/checkpoints/visdial/{folder_name}')


# Log the model state_dict with best ndcg and best mean
best_mean_epoch = checkpoint_manager.best_mean_epoch
best_ndcg_epoch = checkpoint_manager.best_ndcg_epoch

logging.info(f'save best mean epoch: {best_mean_epoch}')
experiment.log_asset(os.path.join(config['callbacks']['path_dir_save_ckpt'], f"checkpoint_{best_mean_epoch}.pth"))
logging.info(f'save best ndcg epoch: {best_ndcg_epoch}')
experiment.log_asset(os.path.join(config['callbacks']['path_dir_save_ckpt'], f"checkpoint_{best_ndcg_epoch}.pth"))

summary_writer.close()
