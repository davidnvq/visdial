from comet_ml import Experiment, OfflineExperiment

import os
import sys
import torch
import random
import logging
from tqdm import tqdm
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from visdial.model import get_model
from visdial.data.dataset import VisDialDataset
from visdial.metrics import SparseGTMetrics, NDCG
from visdial.utils.checkpointing import CheckpointManager, load_checkpoint_from_config
from visdial.utils import move_to_cuda
from options16 import get_comet_experiment, get_training_config_and_args
from torch.utils.tensorboard import SummaryWriter
from visdial.optim import Adam, LRScheduler, get_weight_decay_params

# Load config
config, args = get_training_config_and_args()
# import yaml
# config_yml = 'configs/overfit_50epoch_lseps_0.2.yml'
# config = yaml.load(open(config_yml),Loader=yaml.SafeLoader)

# Set logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

# Set comet
experiment = get_comet_experiment(config)
device = torch.device('cuda')

# For reproducibility
seed = config['seed']

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(seed)


# DATASET INIT
logging.info("Loading train and validation dataset...")
trainset = VisDialDataset(config, split='train')
train_dl = DataLoader(trainset,
                      batch_size=config['solver']['batch_size'] * torch.cuda.device_count(),
                      num_workers=config['solver']['cpu_workers'],
                      shuffle=True, pin_memory=True)
validset = VisDialDataset(config, split='val')
valid_dl = DataLoader(validset,
                      batch_size=2 * torch.cuda.device_count(),
                      num_workers=config['solver']['cpu_workers'], pin_memory=True)


# MODEL
logging.info("Init model...")
model = get_model(config)
model = model.to(device)

"""LOSS FUNCTION"""
from visdial.loss import VisdialLoss

disc_criterion = VisdialLoss(LS_epsilon=config['solver']['ls_epsilon'], return_mean=True)
gen_criterion = nn.CrossEntropyLoss(ignore_index=0)

"""OPTIMIZER"""
parameters = get_weight_decay_params(model, weight_decay=config['solver']['weight_decay'])
optimizer = Adam(parameters,
                 betas=config['solver']['adam_betas'],
                 eps=config['solver']['adam_eps'],
                 weight_decay=config['solver']['weight_decay'])

lr_scheduler = LRScheduler(optimizer, **config['solver'])

# =============================================================================
#   SETUP BEFORE TRAINING LOOP
# =============================================================================
summary_writer = SummaryWriter(log_dir=config['callbacks']['log_dir'])
checkpoint_manager = CheckpointManager(model, optimizer, config['callbacks']['save_dir'], config=config)
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

iterations = len(trainset) // (config['solver']['batch_size'] * torch.cuda.device_count()) + 1
num_examples = torch.tensor(len(trainset), dtype=torch.float)
global_iteration_step = start_epoch * iterations

for epoch in range(start_epoch, config['solver']['num_epochs']):
    logging.info(f"Training for epoch {epoch}:")
    print(f"Training for epoch {epoch}:")

    with tqdm(total=iterations) as pbar:
        epoch_loss = torch.tensor(0.0)
        for i, batch in enumerate(train_dl):
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
                scores = out['ans_out_scores'].view(-1, config['model']['txt_vocab_size'])
                target = batch['ans_out'].view(-1)
                gen_loss = gen_criterion(scores, target)
                batch_loss = batch_loss + gen_loss

            # compute gradients
            batch_loss.backward()

            # update params
            lr = lr_scheduler.step(global_iteration_step)
            optimizer.step()

            pbar.update(1)

            if global_iteration_step % 200 == 0:
                logging.info("epoch={:02d}, steps={:03d}K: batch_loss:{:.03f} disc_loss:{:.03f} gen_loss:{:.03f} lr={:.05f}".format(
                    epoch, int(global_iteration_step / 200), batch_loss.item(), disc_loss.item(), gen_loss.item(), lr))

            # log metrics
            experiment.log_metric('train/batch_loss', batch_loss.item())
            summary_writer.add_scalar(config['config_name'] + "-train/batch_loss", batch_loss.item(), global_iteration_step)
            experiment.log_metric('train/lr', lr)
            summary_writer.add_scalar("train/batch_lr", lr, global_iteration_step)

            global_iteration_step += 1
            torch.cuda.empty_cache()

            epoch_loss += batch["ans"].size(0) * batch_loss.detach()

        if out.get('opt_scores') is not None:
            avg_metric_dict = {}
            avg_metric_dict.update(sparse_metrics.retrieve(reset=True))

            for metric_name, metric_value in avg_metric_dict.items():
                logging.info(f"{metric_name}: {metric_value}")
                experiment.log_metric(f"train/{metric_name}", metric_value)

            summary_writer.add_scalars(config['config_name'] + "-train/metrics", avg_metric_dict, global_iteration_step)

        epoch_loss /= num_examples
        experiment.log_metric('train/epoch_loss', epoch_loss.item())
        logging.info(f"train/epoch_loss: {epoch_loss.item()}\n")
        summary_writer.add_scalar(config['config_name'] + "-train/epoch_loss", epoch_loss.item(), global_iteration_step)

        for name, param in model.named_parameters():
            summary_writer.add_histogram(name, param.clone().cpu().data.numpy(), global_iteration_step)

    # -------------------------------------------------------------------------
    #   ON EPOCH END  (checkpointing and validation)
    # -------------------------------------------------------------------------
    # Validate and report automatic metrics.

    if config['callbacks']['validate'] and epoch % 5 == 0:
        # Switch dropout, batchnorm etc to the correct mode.
        model.eval()

        logging.info(f"\nValidation after epoch {epoch}:")

        for batch in valid_dl:
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

            summary_writer.add_scalars(config['config_name'] + "-val/metrics", metric_dict, global_iteration_step)

        model.train()
        torch.cuda.empty_cache()

        # Checkpoint
        if 'disc' in config['model']['decoder_type']:
            checkpoint_manager.step(epoch=epoch, only_best=False, metrics=disc_metric_dict, key='disc_')

        elif 'gen' in config['model']['decoder_type']:
            checkpoint_manager.step(epoch=epoch, only_best=False, metrics=gen_metric_dict, key='gen_')

        elif 'misc' in config['model']['decoder_type']:
            checkpoint_manager.step(epoch=epoch, only_best=False, metrics=disc_metric_dict, key='disc_')

summary_writer.close()

# Log the model state_dict with best ndcg and best mean
best_mean_epoch = checkpoint_manager.best_mean_epoch
best_ndcg_epoch = checkpoint_manager.best_ndcg_epoch

logging.info(f'save best mean epoch: {best_mean_epoch}')
logging.info(os.path.join(config['callbacks']['save_dir'], f"checkpoint_{best_mean_epoch}.pth"))

logging.info(f'save best ndcg epoch: {best_ndcg_epoch}')
logging.info(os.path.join(config['callbacks']['save_dir'], f"checkpoint_{best_ndcg_epoch}.pth"))

skip_list = [os.path.join(config['callbacks']['save_dir'], f"checkpoint_{best_mean_epoch}.pth"),
             os.path.join(config['callbacks']['save_dir'], f"checkpoint_{best_ndcg_epoch}.pth"),
             os.path.join(config['callbacks']['save_dir'], "checkpoint_last.pth")]

from glob import glob

all_checkpoints = glob(config['callbacks']['save_dir'] + '/*.pth')
for ckpt_path in all_checkpoints:
    if 'last' in ckpt_path or ckpt_path in skip_list:
        continue
    else:
        os.system(f"rm {ckpt_path}")
