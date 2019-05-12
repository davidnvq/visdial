from comet_ml import Experiment

import os
import yaml
import torch
import random
import argparse
import itertools
import numpy as np

from tqdm import tqdm
from bisect import bisect
from torch import nn, optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from visdial.encoders import Encoder
from visdial.decoders import Decoder
from visdial.model import EncoderDecoderModel
from visdial.data.dataset import VisDialDataset
from visdial.metrics import SparseGTMetrics, NDCG, Monitor
from visdial.utils.checkpointing import CheckpointManager, load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument(
		"--seed",
		type=int,
		default=0,
		help="For reproducibility",
		)
parser.add_argument(
		"--config-yml",
		default="configs/lf_disc_faster_rcnn_x101.yml",
		help="Path to a config file listing reader, model and solver parameters.",
		)
parser.add_argument(
		"--comet-name",
		default="test",
		help="Path to a config file listing reader, model and solver parameters.",
		)
parser.add_argument(
		"--train-json",
		default="data/visdial_1.0_train.json",
		help="Path to json file containing VisDial v1.0 training data.",
		)
parser.add_argument(
		"--val-json",
		default="data/visdial_1.0_val.json",
		help="Path to json file containing VisDial v1.0 validation data.",
		)

parser.add_argument(
		"--lr",
		default=1e-3,
		type=float,
		help="lr",
		)

parser.add_argument(
		"--val-dense-json",
		default="data/visdial_1.0_val_dense_annotations.json",
		help="Path to json file containing VisDial v1.0 validation dense ground "
		     "truth annotations.",
		)
parser.add_argument_group(
		"Arguments independent of experiment reproducibility"
		)
parser.add_argument(
		"--gpu-ids",
		nargs="+",
		type=int,
		default=0,
		help="List of ids of GPUs to use.",
		)
parser.add_argument(
		"--cpu-workers",
		type=int,
		default=4,
		help="Number of CPU workers for dataloader.",
		)
parser.add_argument(
		"--overfit",
		action="store_true",
		help="Overfit model on 5 examples, meant for debugging.",
		)
parser.add_argument(
		"--validate",
		action="store_true",
		help="Whether to validate on val split after every epoch.",
		)
parser.add_argument(
		"--in-memory",
		action="store_true",
		help="Load the whole dataset and pre-extracted image features in memory. "
		     "Use only in presence of large RAM, atleast few tens of GBs.",
		)

parser.add_argument_group("Checkpointing related arguments")
parser.add_argument(
		"--save-dirpath",
		default="checkpoints/",
		help="Path of directory to create checkpoint directory and save "
		     "checkpoints.",
		)
parser.add_argument(
		"--monitor-path",
		default="data/lf_disc_val.pkl",
		help="Path of directory to create checkpoint directory and save "
		     "checkpoints.",
		)

parser.add_argument(
		"--load-pthpath",
		default="",
		help="To continue training, path to .pth file of saved checkpoint.",
		)

# =============================================================================
#   INPUT ARGUMENTS AND CONFIG
# =============================================================================

args = parser.parse_args()

# keys: {"dataset", "model", "solver"}
config = yaml.load(open(args.config_yml))

if isinstance(args.gpu_ids, int):
	args.gpu_ids = [args.gpu_ids]

if len(args.gpu_ids) == 1:
	device = torch.device('cuda')
elif len(args.gpu_ids) > 1:
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

# Print config and args.
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
	print("{:<20}: {}".format(arg, getattr(args, arg)))

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
		args.train_json,
		overfit=args.overfit,
		in_memory=args.in_memory,
		return_options=is_return,
		add_boundary_toks=is_abtoks,
		)
train_dataloader = DataLoader(
		train_dataset,
		batch_size=config["solver"]["batch_size"],
		num_workers=args.cpu_workers,
		shuffle=True,
		)

val_dataset = VisDialDataset(
		config["dataset"],
		args.val_json,
		args.val_dense_json,
		overfit=args.overfit,
		in_memory=args.in_memory,
		return_options=True,
		add_boundary_toks=is_abtoks,
		)

val_dataloader = DataLoader(
		val_dataset,
		batch_size=config["solver"]["batch_size"]
		if config["model"]["decoder"] == "disc"
		else 5,
		num_workers=args.cpu_workers,
		worker_init_fn=init_fn
		)

# TODO: Uncomment this overfit test
# train_dataset = val_dataset
# train_dataloader = val_dataloader

# Pass vocabulary to construct Embedding layer.
encoder = Encoder(config["model"], train_dataset.vocabulary)
decoder = Decoder(config["model"], train_dataset.vocabulary)
print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))

# Share word embedding between encoder and decoder.
decoder.word_embed = encoder.word_embed

is_bilstm = config['model']['is_bilstm']

# Wrap encoder and decoder in a model.
model = EncoderDecoderModel(encoder, decoder, is_bilstm=is_bilstm)

model = model.to(device)

if len(args.gpu_ids) > 1:
	model = nn.DataParallel(model)

# Loss function.
if config["model"]["decoder"] == "disc":
	criterion = nn.CrossEntropyLoss()
elif config["model"]["decoder"] == "gen":
	criterion = nn.CrossEntropyLoss(
			ignore_index=train_dataset.vocabulary.PAD_INDEX
			)
else:
	raise NotImplementedError

if config["solver"]["training_splits"] == "trainval":
	iterations = (len(train_dataset) + len(val_dataset)) // config["solver"][
		"batch_size"
	] + 1
	num_examples = torch.tensor(len(train_dataset) + len(val_dataset), dtype=torch.float)
else:
	iterations = len(train_dataset) // config["solver"]["batch_size"] + 1
	num_examples = torch.tensor(len(train_dataset), dtype=torch.float)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

# =============================================================================
#   SETUP BEFORE TRAINING LOOP
# =============================================================================
summary_writer = SummaryWriter(log_dir=args.save_dirpath)
checkpoint_manager = CheckpointManager(
		model, optimizer, args.save_dirpath, config=config
		)

sparse_metrics = SparseGTMetrics()
ndcg = NDCG()
monitor = Monitor(val_dataset, save_path=args.monitor_path)

# If loading from checkpoint, adjust start epoch and load parameters.
if args.load_pthpath == "":
	start_epoch = 0
else:
	# "path/to/checkpoint_xx.pth" -> xx
	start_epoch = int(args.load_pthpath.split("_")[-1][:-4]) + 1

	model_state_dict, optimizer_state_dict = load_checkpoint(args.load_pthpath)
	if isinstance(model, nn.DataParallel):
		model.module.load_state_dict(model_state_dict)
	else:
		model.load_state_dict(model_state_dict)
	optimizer.load_state_dict(optimizer_state_dict)
	print("Loaded model from {}".format(args.load_pthpath))

# =============================================================================
#   TRAINING LOOP
# =============================================================================

# Forever increasing counter to keep track of iterations (for tensorboard log).
global_iteration_step = start_epoch * iterations

for epoch in range(start_epoch, config["solver"]["num_epochs"]):

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

		torch.cuda.empty_cache()
		for i, batch in enumerate(combined_dataloader):
			for key in batch:
				batch[key] = batch[key].to(device)

			optimizer.zero_grad()
			output = model(batch)

			if config["model"]["decoder"] == "disc":
				sparse_metrics.observe(output, batch["ans_ind"])

			target = (
				batch["ans_ind"]
				if config["model"]["decoder"] == "disc"
				else batch["ans_out"]
			)

			# compute loss
			batch_loss = criterion(
					output.view(-1, output.size(-1)), target.view(-1)
					)

			# compute gradients
			batch_loss.backward()

			# update params
			optimizer.step()

			summary_writer.add_scalar(
					"train/loss", batch_loss, global_iteration_step
					)
			summary_writer.add_scalar(
					"train/lr", optimizer.param_groups[0]["lr"], global_iteration_step
					)

			# scheduler.step(global_iteration_step)
			global_iteration_step += 1
			torch.cuda.empty_cache()

			pbar.set_postfix(epoch=epoch, batch_loss=batch_loss.item())
			pbar.update(1)

			experiment.log_metric('train/batch_loss', batch_loss.item())
			epoch_loss += batch["ques"].size(0) * batch_loss.detach()

		if config["model"]["decoder"] == "disc":
			all_metrics = {}
			all_metrics.update(sparse_metrics.retrieve(reset=True))

			for metric_name, metric_value in all_metrics.items():
				print(f"{metric_name}: {metric_value}")
				experiment.log_metric(f"train/{metric_name}", metric_value)
			summary_writer.add_scalars(
					"train/metrics", all_metrics, global_iteration_step
					)

		epoch_loss /= num_examples

		summary_writer.add_scalar(
				"train/epoch_loss", epoch_loss, i
				)
		experiment.log_metric('train/epoch_loss', epoch_loss.item())
	# -------------------------------------------------------------------------
	#   ON EPOCH END  (checkpointing and validation)
	# -------------------------------------------------------------------------
	checkpoint_manager.step(epoch=epoch)

	# Validate and report automatic metrics.
	if args.validate:

		# Switch dropout, batchnorm etc to the correct mode.
		model.eval()

		epoch_loss = torch.tensor(0.0)
		print(f"\nValidation after epoch {epoch}:")
		for i, batch in enumerate(tqdm(val_dataloader)):
			for key in batch:
				batch[key] = batch[key].to(device)

			with torch.no_grad():
				output, attn_weights = model(batch, debug=True)

			sparse_metrics.observe(output, batch["ans_ind"])

			monitor.update(batch['img_ids'].detach(),
			               output.detach(),
			               batch['ans_ind'].detach(),
			               attn_weights.detach())

			if "gt_relevance" in batch:
				output = output[
				         torch.arange(output.size(0)), batch["round_id"] - 1, :
				         ]
				ndcg.observe(output, batch["gt_relevance"])

		monitor.export()

		all_metrics = {}
		all_metrics.update(sparse_metrics.retrieve(reset=True))
		all_metrics.update(ndcg.retrieve(reset=True))

		for metric_name, metric_value in all_metrics.items():
			print(f"{metric_name}: {metric_value}")
			experiment.log_metric(f"val/{metric_name}", metric_value)

		summary_writer.add_scalars(
				"val/metrics", all_metrics, global_iteration_step
				)

		model.train()
		torch.cuda.empty_cache()
