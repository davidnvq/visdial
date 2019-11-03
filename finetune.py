# from comet_ml import Experiment, OfflineExperiment
# ssh zao/ su - administrator
# sudo qmod -c "*"
import os
import sys
import csv
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
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
from torch.utils.tensorboard import SummaryWriter
from visdial.optim import Adam, LRScheduler, get_weight_decay_params
from visdial.loss import FinetuneLoss
import argparse
import yaml
import pickle
import json
from tqdm import tqdm
import itertools

# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/v002_abc_LP_lkf_D36.yml')
parser.add_argument('--path_pretrained_ckpt', default='')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--init_lr', type=float, default=5e-5)
parser.add_argument('--scheduler_type', type=str, default='CosineLR')
parser.add_argument('--batch_size', type=int, default=8)

args = parser.parse_args()
if 'yml' in args.config:
    config = yaml.load(open(args.config),Loader=yaml.SafeLoader)
elif 'json' in args.config:
    with open(args.config) as file:
        config = json.load(file)

config['dataset']['train_json_dense_dialog_path'] = '/media/local_workspace/quang/datasets/visdial/annotations/visdial_1.0_train_dense_sample.json'
config['dataset']['finetune'] = True
config['callbacks']['path_pretrained_ckpt'] = args.path_pretrained_ckpt
config['solver']['num_epochs'] = args.num_epochs

# Set logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


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

logging.info("Loading val dataset...")
print("Loading val dataset...")

# config['dataset']['overfit'] = True

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

finetune_path = os.path.join(os.path.dirname(config['callbacks']['path_pretrained_ckpt']), 'finetune', f'lr_{str(init_lr)}', scheduler_type)
if not os.path.exists(finetune_path):
    os.makedirs(finetune_path)
print(finetune_path)

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

# LOG TO CSV file
csv_file_path1 = os.path.join(finetune_path, "finetune.csv")
hparams = {}
hparams['model'] = config['config_name']
hparams['model_epoch'] = args.path_pretrained_ckpt.split("/")[-1]
hparams['init_lr'] = args.init_lr
hparams['scheduler_type'] = args.scheduler_type
hparams['batch_size'] = args.batch_size

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
    logging.info(f"Training for epoch {epoch}:")

    hparams['epoch'] = epoch

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
            # experiment.log_metric('train/batch_loss', batch_loss.item())
            summary_writer.add_scalar(f'{config["config_name"]}-train/batch_loss', batch_loss.item(), global_iteration_step)

            # experiment.log_metric('train/lr', lr)
            summary_writer.add_scalar("train/batch_lr", lr, global_iteration_step)

            global_iteration_step += 1
            torch.cuda.empty_cache()

            epoch_loss += batch["ans"].size(0) * batch_loss.detach()

    if out.get('opt_scores') is not None:
        avg_metric_dict = {}
        avg_metric_dict.update(sparse_metrics.retrieve(reset=True))

        for metric_name, metric_value in avg_metric_dict.items():
            logging.info(f"{metric_name}: {metric_value}")
            # experiment.log_metric(f"train/{metric_name}", metric_value)
            hparams[f"train/{metric_name}"] = metric_value

        summary_writer.add_scalars(f"{config['config_name']}-train/metrics", avg_metric_dict, global_iteration_step)

    epoch_loss /= num_examples
    # experiment.log_metric('train/epoch_loss', epoch_loss.item())
    logging.info(f"train/epoch_loss: {epoch_loss.item()}\n")
    summary_writer.add_scalar(f'{config["config_name"]}-train/epoch_loss', epoch_loss.item(), global_iteration_step)
    hparams[f"train/epoch_loss"] = epoch_loss.item()

    # -------------------------------------------------------------------------
    #   ON EPOCH END  (checkpointing and validation)
    # -------------------------------------------------------------------------
    # Validate and report automatic metrics.

    if True:
        # Switch dropout, batchnorm etc to the correct mode.
        model.eval()

        logging.info(f"\nValidation after epoch {epoch}:")

        all_disc_outputs = []
        all_misc_outputs = []
        all_gen_outputs = []
        all_img_ids = []
        all_round_ids = []

        num_imgs = 0

        for batch in tqdm(eval_dataloader):
            all_img_ids.append(batch['img_ids'])
            all_round_ids.append(batch['round_id'])

            torch.cuda.empty_cache()

            move_to_cuda(batch, device)

            with torch.no_grad():
                out = model(batch)

                if out.get('opt_scores') is not None:
                    scores = out['opt_scores']
                    disc_metrics.observe(scores, batch["ans_ind"])
                    all_disc_outputs.append((torch.softmax(scores, dim=-1)).cpu())

                    if "gt_relevance" in batch:
                        scores = scores[
                                 torch.arange(scores.size(0)),
                                 batch["round_id"] - 1, :]

                        disc_ndcg.observe(scores, batch["gt_relevance"])

                if out.get('opts_out_scores') is not None:
                    scores = out['opts_out_scores']
                    all_gen_outputs.append((torch.softmax(scores, dim=-1)).cpu())

                    gen_metrics.observe(scores, batch["ans_ind"])

                    if "gt_relevance" in batch:
                        scores = scores[
                                 torch.arange(scores.size(0)),
                                 batch["round_id"] - 1, :]

                        gen_ndcg.observe(scores, batch["gt_relevance"])

                if out.get('opt_scores') is not None and out.get('opts_out_scores') is not None:
                    scores = (out['opts_out_scores'] + out['opt_scores']) / 2

                    all_misc_outputs.append(((torch.softmax(out['opt_scores'], dim=-1) +
                                             torch.softmax(out['opts_out_scores'], dim=-1)) / 2.0).cpu())

                    sparse_metrics.observe(scores, batch["ans_ind"])
                    if "gt_relevance" in batch:
                        scores = scores[
                                 torch.arange(scores.size(0)),
                                 batch["round_id"] - 1, :]

                        ndcg.observe(scores, batch["gt_relevance"])

        rank_path = os.path.join(finetune_path, 'ranks', 'val', f'ckpt_{epoch}')
        if not os.path.exists(rank_path):
            os.makedirs(rank_path)

        model_name = os.path.basename(args.path_pretrained_ckpt).split()[0]
        disc_path = os.path.join(rank_path, f'{model_name}_disc.pkl')
        misc_path = os.path.join(rank_path, f'{model_name}_misc.pkl')
        gen_path = os.path.join(rank_path, f'{model_name}_gen.pkl')

        with open(disc_path, 'wb') as f:
            pickle.dump([all_disc_outputs, all_img_ids, all_round_ids], f)

        with open(misc_path, 'wb') as f:
            pickle.dump([all_misc_outputs, all_img_ids, all_round_ids], f)

        with open(gen_path, 'wb') as f:
            pickle.dump([all_gen_outputs, all_img_ids, all_round_ids], f)

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
                # experiment.log_metric(f"val/{metric_name}", metric_value)
                hparams[f"val/{metric_name}"] = metric_value

            summary_writer.add_scalars(f"{config['config_name']}-val/metrics", metric_dict, global_iteration_step)

        csv_columns = list(hparams.keys())
        if not os.path.exists(csv_file_path1):
            with open(csv_file_path1, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()

        with open(csv_file_path1, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writerow(hparams)


        # # Switch dropout, batchnorm etc to the correct mode.
        # logging.info(f"\nTest after epoch {epoch}:")
        #
        # all_disc_outputs = []
        # all_misc_outputs = []
        # all_gen_outputs = []
        # all_img_ids = []
        # all_round_ids = []
        #
        # for batch in tqdm(test_dataloader):
        #     all_img_ids.append(batch['img_ids'])
        #     all_round_ids.append(batch['img_ids'])
        #
        #     move_to_cuda(batch, device)
        #
        #     with torch.no_grad():
        #         out = model(batch, test_mode=True)
        #
        #         if out.get('opt_scores') is not None:
        #             scores = out['opt_scores']
        #             all_disc_outputs.append((torch.softmax(scores, dim=-1)).cpu())
        #
        #         if out.get('opts_out_scores') is not None:
        #             scores = out['opts_out_scores']
        #             all_gen_outputs.append((torch.softmax(scores, dim=-1)).cpu())
        #
        #         if out.get('opt_scores') is not None and out.get('opts_out_scores') is not None:
        #             scores = (out['opts_out_scores'] + out['opt_scores']) / 2
        #             all_misc_outputs.append(((torch.softmax(out['opt_scores'], dim=-1) +
        #                                       torch.softmax(out['opts_out_scores'], dim=-1)) / 2.0).cpu())
        #
        # rank_path = os.path.join(finetune_path, 'ranks', 'test', f'ckpt_{epoch}')
        # if not os.path.exists(rank_path):
        #     os.makedirs(rank_path)
        #
        # model_name = os.path.basename(args.path_pretrained_ckpt).split()[0]
        # disc_path = os.path.join(rank_path, f'{model_name}_disc.pkl')
        # misc_path = os.path.join(rank_path, f'{model_name}_misc.pkl')
        # gen_path = os.path.join(rank_path, f'{model_name}_gen.pkl')
        #
        # with open(disc_path, 'wb') as f:
        #     pickle.dump([all_disc_outputs, all_img_ids, all_round_ids], f)
        #
        # with open(misc_path, 'wb') as f:
        #     pickle.dump([all_misc_outputs, all_img_ids, all_round_ids], f)
        #
        # with open(gen_path, 'wb') as f:
        #     pickle.dump([all_gen_outputs, all_img_ids, all_round_ids], f)

        model.train()
        torch.cuda.empty_cache()

        # Checkpoint
        checkpoint_manager.step(epoch=epoch, only_best=False, metrics=disc_metric_dict, key='disc_')

    if epoch == 5:
        break

# Log the model state_dict with best ndcg and best mean
best_mean_epoch = checkpoint_manager.best_mean_epoch
best_ndcg_epoch = checkpoint_manager.best_ndcg_epoch

logging.info(f'save best mean epoch: {best_mean_epoch}')
logging.info(f'save best ndcg epoch: {best_ndcg_epoch}')

summary_writer.add_scalar('best_mean_epoch', best_mean_epoch)
summary_writer.add_scalar('best_ndcg_epoch', best_ndcg_epoch)
summary_writer.close()
print(finetune_path)