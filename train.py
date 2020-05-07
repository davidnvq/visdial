import os
import torch
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from visdial.model import get_model
from torch.utils.data import DataLoader
from visdial.data.dataset import VisDialDataset
from visdial.metrics import SparseGTMetrics, NDCG
from visdial.utils.checkpointing import CheckpointManager, load_checkpoint_from_config
from visdial.utils import move_to_cuda
from visdial.common.utils import check_flag
from options import get_training_config_and_args
from torch.utils.tensorboard import SummaryWriter
from visdial.optim import Adam, LRScheduler, get_weight_decay_params

config, args = get_training_config_and_args()

seed = config['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(seed)

print(f"CUDA number: {torch.cuda.device_count()}")

"""DATASET INIT"""
print("Loading val dataset...")
val_dataset = VisDialDataset(config, split='val')

if check_flag(config['dataset'], 'v0.9'):
    val_dataset.dense_ann_feat_reader = None

val_dataloader = DataLoader(val_dataset,
                            batch_size=config['solver']['batch_size'] / 2 * torch.cuda.device_count(),
                            num_workers=config['solver']['cpu_workers'])

print("Loading train dataset...")
if config['dataset']['overfit']:
    train_dataset = val_dataset
    train_dataloader = val_dataloader
else:
    train_dataset = VisDialDataset(config, split='train')
    if check_flag(config['dataset'], 'v0.9'):
        train_dataset.dense_ann_feat_reader = None

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['solver']['batch_size'] * torch.cuda.device_count(),
                                  num_workers=config['solver']['cpu_workers'],
                                  shuffle=True)

"""MODEL INIT"""
print("Init model...")
device = torch.device('cuda')
model = get_model(config)
model = model.to(device)

"""LOSS FUNCTION"""
from visdial.loss import DiscLoss

disc_criterion = DiscLoss(return_mean=True)
gen_criterion = nn.CrossEntropyLoss(ignore_index=0)

"""OPTIMIZER"""
parameters = get_weight_decay_params(model, weight_decay=config['solver']['weight_decay'])

optimizer = Adam(parameters,
                 betas=config['solver']['adam_betas'],
                 eps=config['solver']['adam_eps'],
                 weight_decay=config['solver']['weight_decay'])

lr_scheduler = LRScheduler(optimizer,
                           batch_size=config['solver']['batch_size'] * torch.cuda.device_count(),
                           num_samples=config['solver']['num_samples'],
                           num_epochs=config['solver']['num_epochs'],
                           min_lr=config['solver']['min_lr'],
                           init_lr=config['solver']['init_lr'],
                           warmup_factor=config['solver']['warmup_factor'],
                           warmup_epochs=config['solver']['warmup_epochs'],
                           scheduler_type=config['solver']['scheduler_type'],
                           milestone_steps=config['solver']['milestone_steps'],
                           linear_gama=config['solver']['linear_gama']
                           )

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

print("Loading checkpoints...")
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
    print(f"Training for epoch {epoch}:")
    print(f"Training for epoch {epoch}:")
    if check_flag(config['dataset'], 'v0.9') and epoch > 6:
        break

    epoch_loss = torch.tensor(0.0)
    for batch in tqdm(train_dataloader, total=iterations, unit="batch"):
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

        # logging
        if config['dataset']['overfit']:
            print("epoch={:02d}, steps={:03d}K: batch_loss:{:.03f} "
                  "disc_loss:{:.03f} gen_loss:{:.03f} lr={:.05f}".format(
                epoch, int(global_iteration_step / 1000), batch_loss.item(),
                disc_loss.item(), gen_loss.item(), lr))

        if global_iteration_step % 1000 == 0:
            print("epoch={:02d}, steps={:03d}K: batch_loss:{:.03f} "
                  "disc_loss:{:.03f} gen_loss:{:.03f} lr={:.05f}".format(
                epoch, int(global_iteration_step / 1000), batch_loss.item(),
                disc_loss.item(), gen_loss.item(), lr))

        summary_writer.add_scalar(config['config_name'] + "-train/batch_loss",
                                  batch_loss.item(), global_iteration_step)
        summary_writer.add_scalar("train/batch_lr", lr, global_iteration_step)

        global_iteration_step += 1
        torch.cuda.empty_cache()

        epoch_loss += batch["ans"].size(0) * batch_loss.detach()

    if out.get('opt_scores') is not None:
        avg_metric_dict = {}
        avg_metric_dict.update(sparse_metrics.retrieve(reset=True))

        summary_writer.add_scalars(config['config_name'] + "-train/metrics",
                                   avg_metric_dict, global_iteration_step)

        for metric_name, metric_value in avg_metric_dict.items():
            print(f"{metric_name}: {metric_value}")

    epoch_loss /= num_examples
    summary_writer.add_scalar(config['config_name'] + "-train/epoch_loss",
                              epoch_loss.item(), global_iteration_step)

    # -------------------------------------------------------------------------
    #   ON EPOCH END  (checkpointing and validation)
    # -------------------------------------------------------------------------
    # Validate and report automatic metrics.

    if config['callbacks']['validate']:
        # Switch dropout, batchnorm etc to the correct mode.
        model.eval()

        print(f"\nValidation after epoch {epoch}:")

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
                print(f"{metric_name}: {metric_value}")
            summary_writer.add_scalars(config['config_name'] + "-val/metrics",
                                       metric_dict, global_iteration_step)

        model.train()
        torch.cuda.empty_cache()

        # Checkpoint
        if not args.overfit:
            if 'disc' in config['model']['decoder_type']:
                checkpoint_manager.step(epoch=epoch, only_best=False, metrics=disc_metric_dict, key='disc_')

            elif 'gen' in config['model']['decoder_type']:
                checkpoint_manager.step(epoch=epoch, only_best=False, metrics=gen_metric_dict, key='gen_')

            elif 'misc' in config['model']['decoder_type']:
                checkpoint_manager.step(epoch=epoch, only_best=False, metrics=disc_metric_dict, key='disc_')
