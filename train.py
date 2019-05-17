from comet_ml import Experiment
import os
import sys
import yaml
import torch
import random
import argparse
import itertools
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from visdial.encoders import Encoder
from visdial.decoders import Decoder
from visdial.model import EncoderDecoderModel
from visdial.loss import get_criterion
from visdial.data.dataset import VisDialDataset
from visdial.metrics import SparseGTMetrics, NDCG, Monitor
from visdial.utils.checkpointing import CheckpointManager, load_checkpoint
from visdial.utils import move_to_cuda


parser = argparse.ArgumentParser()
parser.add_argument("--image-features-tr-h5", default="")
parser.add_argument("--image-features-va-h5", default="")
parser.add_argument("--image-features-te-h5", default="")
parser.add_argument("--json-word-counts", default="")
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--num-epochs", default=20, type=int)
parser.add_argument("--batch-size", default=16, type=int)
parser.add_argument("--in-memory", action="store_true")
parser.add_argument("--validate", action="store_true")
parser.add_argument("--overfit", action="store_true")
parser.add_argument("--gpu-ids", nargs="+", default=0, type=int)
parser.add_argument("--lr-steps", nargs="+", default=[5, ], type=int)
parser.add_argument("--cpu-workers", default=4, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--config-yml", default="")
parser.add_argument("--comet-name", default="")
parser.add_argument("--json-train", default="")
parser.add_argument("--json-val", default="")
parser.add_argument("--json-val-dense", default="")
parser.add_argument("--save-dirpath", default="")
parser.add_argument("--device", default="cuda")
parser.add_argument("--step-size", default=10, type=int)
parser.add_argument("--load-pthpath", default="")
parser.add_argument("--resume", action="store_true")

args = parser.parse_args()

# =============================================================================
#   INPUT ARGUMENTS AND CONFIG
# =============================================================================

# keys: {"dataset", "model", "solver"}
with open(args.config_yml, 'r') as file:
	config = yaml.load(file)

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key='2z9VHjswAJWF1TV6x4WcFMVss',
                        project_name=args.comet_name, workspace="lightcv")

experiment.log_asset(args.config_yml)

for key in ['model', 'solver']:
	experiment.log_parameters(config[key])


# =============================================================================
# For reproducibility.
# =============================================================================
# Refer https://pytorch.org/docs/stable/notes/randomness.html

def seed_torch(seed=0):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	os.environ['PYTHONHASHSEED'] = str(seed)

	def worker_init_fn(worker_id):
		np.random.seed(seed + worker_id)

	return worker_init_fn


init_fn = seed_torch(args.seed)

# =============================================================================
#   SETUP DATASET, DATALOADER, MODEL, CRITERION, OPTIMIZER, SCHEDULER
# =============================================================================

is_abtoks = True if config["model"]["decoder"] != "disc" else False
is_return = True if config["model"]["decoder"] == "disc" else False

# TODO: Comment this overfit test

train_dataset = VisDialDataset(
		config["dataset"],
		jsonpath_dialogs=args.json_train,
		hdfpath_img_features=args.image_features_tr_h5,
		jsonpath_vocab=args.json_word_counts,
		overfit=args.overfit,
		return_options=is_return,
		add_boundary_toks=is_abtoks,
		)

train_dataloader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		num_workers=args.cpu_workers,
		shuffle=True,
		worker_init_fn=init_fn
		)

val_dataset = VisDialDataset(
		config["dataset"],
		jsonpath_dialogs=args.json_val,
		jsonpath_vocab=args.json_word_counts,
		jsonpath_dense=args.json_val_dense,
		hdfpath_img_features=args.image_features_va_h5,
		overfit=args.overfit,
		return_options=True,
		add_boundary_toks=is_abtoks,
		)

val_dataloader = DataLoader(
		val_dataset,
		batch_size=4,
		num_workers=args.cpu_workers,
		worker_init_fn=init_fn
		)

# # TODO: Uncomment this overfit test
# train_dataset = val_dataset
# train_dataloader = val_dataloader

# Pass vocabulary to construct Embedding layer.
print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))

encoder = Encoder(config["model"], train_dataset.vocabulary)
decoder = Decoder(config["model"], train_dataset.vocabulary)
# Share word embedding between encoder and decoder.
decoder.word_embed = encoder.word_embed

# Wrap encoder and decoder in a model.
model = EncoderDecoderModel(encoder, decoder)

device = torch.device('cuda')

model = model.to(device)

# Loss function.
criterion = get_criterion(config['model']['decoder'])

if config["solver"]["training_splits"] == "trainval":
	iterations = (len(train_dataset) + len(val_dataset)) // args.batch_size + 1
	num_examples = torch.tensor(len(train_dataset) + len(val_dataset), dtype=torch.float)
else:
	iterations = len(train_dataset) // args.batch_size + 1
	num_examples = torch.tensor(len(train_dataset), dtype=torch.float)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)

# =============================================================================
#   SETUP BEFORE TRAINING LOOP
# =============================================================================
summary_writer = SummaryWriter(log_dir=args.save_dirpath)
checkpoint_manager = CheckpointManager(model, optimizer, args.save_dirpath, config=config)
sparse_metrics = SparseGTMetrics()
ndcg = NDCG()

# monitor = Monitor(val_dataset, save_path=args.monitor_path + '/val_monitor.pkl')

# If loading from checkpoint, adjust start epoch and load parameters.
start_epoch = 0

if args.load_pthpath != "":
	if args.resume:
		start_epoch, model, optimizer = load_checkpoint(
				args.load_pthpath,
				model, optimizer,
				device=args.device, resume=True)
	else:
		model = load_checkpoint(args.load_pthpath, model, device=args.device)

if isinstance(args.gpu_ids, int):
	args.gpu_ids = [args.gpu_ids]

if len(args.gpu_ids) > 1:
	model = nn.DataParallel(model)

# =============================================================================
#   TRAINING LOOP
# =============================================================================
# Forever increasing counter to keep track of iterations (for tensorboard log).
global_iteration_step = start_epoch * iterations


for epoch in range(start_epoch, args.num_epochs):
	# -------------------------------------------------------------------------
	#   ON EPOCH START  (combine dataloaders if training on train + val)
	# -------------------------------------------------------------------------
	if config["solver"]["training_splits"] == "trainval":
		combined_dataloader = itertools.chain(train_dataloader, val_dataloader)
	else:
		combined_dataloader = itertools.chain(train_dataloader)

	print(f"\nTraining for epoch {epoch}:")

	with tqdm(total=iterations) as pbar:
		epoch_loss = torch.tensor(0.0)

		for i, batch in enumerate(combined_dataloader):
			batch = move_to_cuda(batch, device)

			# zero grad
			optimizer.zero_grad()

			# do forward
			output = model(batch)

			# get target
			if config["model"]["decoder"] == "disc":
				target = batch['ans_ind']
				sparse_metrics.observe(output, target)
			else:
				target = batch["ans_out"]

			# compute loss
			batch_loss = criterion(output.view(-1, output.size(-1)), target.view(-1))

			# compute gradients
			batch_loss.backward()

			# update params
			optimizer.step()

			pbar.set_postfix(epoch=epoch, batch_loss=batch_loss.item())
			pbar.update(1)

			experiment.log_metric('train/batch_loss', batch_loss.item())
			summary_writer.add_scalar("train/batch_loss", batch_loss, global_iteration_step)
			summary_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_iteration_step)

			global_iteration_step += 1
			torch.cuda.empty_cache()

			epoch_loss += batch["ques"].size(0) * batch_loss.detach()

		if config["model"]["decoder"] == "disc":
			all_metrics = {}
			all_metrics.update(sparse_metrics.retrieve(reset=True))

			for metric_name, metric_value in all_metrics.items():
				print('')
				print(f"{metric_name}: {metric_value}")
				experiment.log_metric(f"train/{metric_name}", metric_value)

			summary_writer.add_scalars("train/metrics", all_metrics, global_iteration_step)

		epoch_loss /= num_examples

		summary_writer.add_scalar("train/epoch_loss", epoch_loss, i)
		experiment.log_metric('train/epoch_loss', epoch_loss.item())

	scheduler.step(epoch)
	# -------------------------------------------------------------------------
	#   ON EPOCH END  (checkpointing and validation)
	# -------------------------------------------------------------------------
	# Validate and report automatic metrics.
	if args.validate:

		# Switch dropout, batchnorm etc to the correct mode.
		model.eval()

		epoch_loss = torch.tensor(0.0)

		print(f"\nValidation after epoch {epoch}:")

		for i, batch in enumerate(tqdm(val_dataloader)):

			move_to_cuda(batch, device)

			with torch.no_grad():
				output = model(batch)
				sparse_metrics.observe(output, batch["ans_ind"])

				if "gt_relevance" in batch:
					output = output[torch.arange(output.size(0)), batch["round_id"] - 1, :]
					ndcg.observe(output, batch["gt_relevance"])

		# monitor.update(batch['img_ids'], output, batch['ans_ind'])

		# if 'gt_relevance' in batch:
		# 	monitor.export()

		all_metrics = {}
		all_metrics.update(sparse_metrics.retrieve(reset=True))
		all_metrics.update(ndcg.retrieve(reset=True))

		for metric_name, metric_value in all_metrics.items():
			print(f"{metric_name}: {metric_value}")
			experiment.log_metric(f"val/{metric_name}", metric_value)

		summary_writer.add_scalars("val/metrics", all_metrics, global_iteration_step)

		model.train()
		torch.cuda.empty_cache()

	# Checkpoint
	checkpoint_manager.step(epoch=epoch, only_best=True, metrics=all_metrics)
