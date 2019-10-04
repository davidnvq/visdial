from comet_ml import Experiment, OfflineExperiment

import os
import sys
import torch
import random
import logging
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from visdial.model import get_model
from visdial.data.dataset import VisDialDataset
from visdial.metrics import SparseGTMetrics, NDCG
from visdial.utils.checkpointing import CheckpointManager, load_checkpoint_from_config
from visdial.utils import move_to_cuda
from options import get_comet_experiment, get_training_config_and_args
from torch.utils.tensorboard import SummaryWriter
from visdial.optim import Adam, LRScheduler, get_weight_decay_params
from visdial.loss import FinetuneLoss
import argparse
import yaml
import json
from tqdm import tqdm
import itertools

# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/v002_abc_LP_lkf_D36.yml')
parser.add_argument('--path_pretrained_ckpt', default='')

args = parser.parse_args()
config = yaml.load(open(args.config),Loader=yaml.SafeLoader)

config['dataset']['train_json_dense_dialog_path'] = '/media/local_workspace/quang/datasets/visdial/annotations/visdial_1.0_train_dense_sample.json'
config['dataset']['finetune'] = True
config['callbacks']['path_pretrained_ckpt'] = args.path_pretrained_ckpt

# Set logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


# Set comet
experiment = Experiment(api_key='2z9VHjswAJWF1TV6x4WcFMVss',
	                  project_name='finetune',
	                  workspace='lightcv')

print(json.dumps(config, indent=2))

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

# datasets
logging.info(f"CUDA number: {torch.cuda.device_count()}")

"""DATASET INIT"""
logging.info("Loading train dataset...")
print("Loading train dataset...")
train_dataset = VisDialDataset(config, split='train')

train_dataloader = DataLoader(train_dataset,
                              batch_size=config['solver']['batch_size'] * torch.cuda.device_count(),
                              num_workers=config['solver']['cpu_workers'],
                              shuffle=True)

logging.info("Loading val dataset...")
print("Loading val dataset...")
val_dataset = VisDialDataset(config, split='val')

val_dataloader = DataLoader(val_dataset,
                            batch_size=config['solver']['batch_size'] * torch.cuda.device_count(),
                            num_workers=config['solver']['cpu_workers'],
                            shuffle=True)

eval_dataloader = DataLoader(val_dataset,
                            batch_size= 4 * torch.cuda.device_count(),
                            num_workers=config['solver']['cpu_workers'],
                            shuffle=False)


"""MODEL INIT"""
logging.info("Init model...")
device = torch.device('cuda')
model = get_model(config)

# load weights
model.load_state_dict(torch.load(config['callbacks']['path_pretrained_ckpt'])['model'])

logging.info("Move model to GPU...")
model = model.to(device)

"""LOSS FUNCTION"""
disc_criterion = FinetuneLoss()


"""OPTIMIZER"""
optimizer = Adam(model.parameters(), lr=5e-5)

lr_scheduler = LRScheduler(optimizer,
                           batch_size=config['solver']['batch_size'] * torch.cuda.device_count(),
                           num_samples=2064 + 2000,
                           num_epochs=config['solver']['num_epochs'],
                           min_lr=1e-5,
                           init_lr=5e-5,
                           warmup_factor=config['solver']['warmup_factor'],
                           warmup_epochs=1,
                           scheduler_type='CosineLR',
                           milestone_steps=config['solver']['milestone_steps'],
                           linear_gama=config['solver']['linear_gama']
                           )

# =============================================================================
#   SETUP BEFORE TRAINING LOOP
# =============================================================================
finetune_path = os.path.dirname(config['callbacks']['path_pretrained_ckpt']) + '/finetune'
if not os.path.exists(finetune_path):
    os.makedirs(finetune_path)

summary_writer = SummaryWriter(log_dir=finetune_path)
checkpoint_manager = CheckpointManager(model, optimizer, finetune_path, config=config)
sparse_metrics = SparseGTMetrics()
disc_metrics = SparseGTMetrics()
gen_metrics = SparseGTMetrics()
ndcg = NDCG()
disc_ndcg = NDCG()
gen_ndcg = NDCG()


if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# =============================================================================
#   TRAINING LOOP
# =============================================================================

config["solver"]["training_splits"] = 'trainval'

start_epoch = 0
if config["solver"]["training_splits"] == "trainval":
    iterations = (len(train_dataset) + len(val_dataset)) // (
				torch.cuda.device_count() * config["solver"]["batch_size"]) + 1
    num_examples = torch.tensor(len(train_dataset) + len(val_dataset), dtype=torch.float)
else:
    iterations = len(train_dataset) // (config['solver']['batch_size'] * torch.cuda.device_count()) + 1
    num_examples = torch.tensor(len(train_dataset), dtype=torch.float)

global_iteration_step = start_epoch * iterations

for epoch in range(start_epoch, config['solver']['num_epochs']):
    logging.info(f"Training for epoch {epoch}:")

    with tqdm(total=iterations) as pbar:
        if config["solver"]["training_splits"] == "trainval":
            combined_dataloader = itertools.chain(train_dataloader, val_dataloader)
        else:
            combined_dataloader = itertools.chain(train_dataloader)

        epoch_loss = torch.tensor(0.0)
        for i, batch in enumerate(combined_dataloader):
            batch = move_to_cuda(batch, device)

            # zero out gradients
            lr = lr_scheduler.step(global_iteration_step)
            optimizer.zero_grad()

            # do forward
            out = model(batch)

            # compute loss
            batch_loss = torch.tensor(0.0, requires_grad=True, device='cuda')
            if out.get('opt_scores') is not None:
                scores = out['opt_scores']

                sparse_metrics.observe(out['opt_scores'], batch['ans_ind'])
                batch_loss = disc_criterion(scores, batch)

            # compute gradients
            batch_loss.backward()

            # update params
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix(epoch=epoch,
                             batch_loss=batch_loss.item())

            # log metrics
            experiment.log_metric('train/batch_loss', batch_loss.item())
            summary_writer.add_scalar(f'{args.config}-train/batch_loss', batch_loss.item(), global_iteration_step)

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

        summary_writer.add_scalars(f"{args.config}-train/metrics", avg_metric_dict, global_iteration_step)

    epoch_loss /= num_examples
    experiment.log_metric('train/epoch_loss', epoch_loss.item())
    logging.info(f"train/epoch_loss: {epoch_loss.item()}\n")
    summary_writer.add_scalar(f'{args.config}-train/epoch_loss', epoch_loss.item(), global_iteration_step)

    # -------------------------------------------------------------------------
    #   ON EPOCH END  (checkpointing and validation)
    # -------------------------------------------------------------------------
    # Validate and report automatic metrics.

    if config['callbacks']['validate']:
        # Switch dropout, batchnorm etc to the correct mode.
        model.eval()

        logging.info(f"\nValidation after epoch {epoch}:")

        for batch in tqdm(eval_dataloader):
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
        checkpoint_manager.step(epoch=epoch, only_best=False, metrics=disc_metric_dict, key='ft_')

# Log the model state_dict with best ndcg and best mean
best_mean_epoch = checkpoint_manager.best_mean_epoch
best_ndcg_epoch = checkpoint_manager.best_ndcg_epoch

logging.info(f'save best mean epoch: {best_mean_epoch}')
logging.info(f'save best ndcg epoch: {best_ndcg_epoch}')

summary_writer.add_scalar('best_mean_epoch', best_mean_epoch)
summary_writer.add_scalar('best_ndcg_epoch', best_ndcg_epoch)
summary_writer.close()
print(finetune_path)