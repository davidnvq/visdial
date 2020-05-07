import os
import sys
import csv
import torch
import random
import logging
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from visdial.model import get_model
from visdial.data.dataset import VisDialDataset
from visdial.metrics import SparseGTMetrics, NDCG
from visdial.utils.checkpointing import CheckpointManager
from visdial.utils import move_to_cuda
from torch.utils.tensorboard import SummaryWriter
from visdial.optim import Adam, LRScheduler
from visdial.loss import FinetuneLoss
import argparse
from tqdm import tqdm
import itertools

# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='checkpoints/model_v1.pth')
parser.add_argument('--save_path', default='checkpoints/finetune')
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--init_lr', type=float, default=1e-4)
parser.add_argument('--scheduler_type', type=str, default='CosineLR')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--overfit', action="store_true", default=False)

args = parser.parse_args()
config_path = os.path.expanduser(args.cpath)
model = torch.load(args.model_path)
config = model.encoder.config

config['dataset']['train_json_dense_dialog_path'] = 'datasets/annotations/visdial_1.0_train_dense_sample.json'
config['dataset']['overfit'] = args.overfit
config['dataset']['finetune'] = True
config['dataset']['evaluate'] = False
config['solver']['num_epochs'] = args.num_epochs

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
print(f"CUDA number: {torch.cuda.device_count()}")

"""DATASET INIT"""
print("Loading dataset...")
val_dataset = VisDialDataset(config, split='val')

val_dataloader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=config['solver']['cpu_workers'],
                            shuffle=True)

if config['dataset']['overfit']:
    train_dataset = val_dataset
    train_dataloader = val_dataloader

train_dataset = VisDialDataset(config, split='train')

train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=config['solver']['cpu_workers'],
                              shuffle=True)

eval_dataset = VisDialDataset(config, split='val')

eval_dataloader = DataLoader(eval_dataset,
                             batch_size=2,
                             num_workers=config['solver']['cpu_workers'])

"""MODEL INIT"""

print("Move model to GPU...")
device = torch.device('cuda')
model = model.to(device)

"""LOSS FUNCTION"""
disc_criterion = FinetuneLoss()

"""OPTIMIZER"""
optimizer = Adam(model.parameters(), lr=2e-5)
init_lr = args.init_lr
scheduler_type = args.scheduler_type
num_epochs = args.num_epochs
lr_scheduler = LRScheduler(optimizer,
                           batch_size=args.batch_size,
                           num_samples=2064 + 2000,
                           num_epochs=args.num_epochs,
                           min_lr=1e-5,
                           init_lr=args.init_lr,
                           warmup_epochs=1,
                           scheduler_type=args.scheduler_type,
                           milestone_steps=config['solver']['milestone_steps'],
                           linear_gama=config['solver']['linear_gama']
                           )

# =============================================================================
#   SETUP BEFORE TRAINING LOOP
# =============================================================================

save_path = args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
print(save_path)

summary_writer = SummaryWriter(log_dir=save_path)
checkpoint_manager = CheckpointManager(model, optimizer, save_path, config=config)
sparse_metrics = SparseGTMetrics()
disc_metrics = SparseGTMetrics()
gen_metrics = SparseGTMetrics()
ndcg = NDCG()
disc_ndcg = NDCG()
gen_ndcg = NDCG()

if torch.cuda.device_count() > 1:
    print("NUMBER OF CUDA", torch.cuda.device_count())
    model = nn.DataParallel(model)

# =============================================================================
#   TRAINING LOOP
# =============================================================================
config["solver"]["training_splits"] = 'trainval'

start_epoch = 0
if config["solver"]["training_splits"] == "trainval":
    iterations = (len(train_dataset) + len(val_dataset)) // (
        args.batch_size) + 1
    num_examples = torch.tensor(len(train_dataset) + len(val_dataset), dtype=torch.float)
else:
    iterations = len(train_dataset) // (args.batch_size) + 1
    num_examples = torch.tensor(len(train_dataset), dtype=torch.float)

global_iteration_step = start_epoch * iterations

for epoch in range(start_epoch, config['solver']['num_epochs']):
    print(f"Training for epoch {epoch}:")

    if epoch == 6:
        break

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
            summary_writer.add_scalar(f'{config["config_name"]}-train/batch_loss',
                                      batch_loss.item(), global_iteration_step)

            # experiment.log_metric('train/lr', lr)
            summary_writer.add_scalar("train/batch_lr", lr, global_iteration_step)

            global_iteration_step += 1
            torch.cuda.empty_cache()

            epoch_loss += batch["ans"].size(0) * batch_loss.detach()

    if out.get('opt_scores') is not None:
        avg_metric_dict = {}
        avg_metric_dict.update(sparse_metrics.retrieve(reset=True))

        for metric_name, metric_value in avg_metric_dict.items():
            print(f"{metric_name}: {metric_value}")

        summary_writer.add_scalars(f"{config['config_name']}-train/metrics",
                                   avg_metric_dict, global_iteration_step)

    epoch_loss /= num_examples
    print(f"train/epoch_loss: {epoch_loss.item()}\n")
    summary_writer.add_scalar(f'{config["config_name"]}-train/epoch_loss',
                              epoch_loss.item(), global_iteration_step)

    # -------------------------------------------------------------------------
    #   ON EPOCH END  (checkpointing and validation)
    # -------------------------------------------------------------------------
    # Validate and report automatic metrics.

    if True:
        # Switch dropout, batchnorm etc to the correct mode.
        model.eval()

        print(f"\nValidation after epoch {epoch}:")

        for batch in tqdm(eval_dataloader):
            torch.cuda.empty_cache()

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
                print(f"{metric_name}: {metric_value}")

            summary_writer.add_scalars(f"{config['config_name']}-val/metrics",
                                       metric_dict, global_iteration_step)

        model.train()
        torch.cuda.empty_cache()

        # Checkpoint
        checkpoint_manager.step(epoch=epoch, only_best=False,
                                metrics=disc_metric_dict, key='disc_')
    if epoch == 5:
        break
