import os
import json
import yaml
import pickle
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


from visdial.encoders import Encoder
from visdial.decoders import Decoder
from visdial.model import EncoderDecoderModel
from visdial.data.dataset import VisDialDataset
from visdial.utils.checkpointing import load_checkpoint
from visdial.metrics import SparseGTMetrics, NDCG, scores_to_ranks


parser = argparse.ArgumentParser(
		"Evaluate and/or generate EvalAI submission file."
		)
parser.add_argument(
		"--config-ymls",
		nargs="+",
		default=['configs/lf_disc_faster_rcnn_x101.yml',
		         'configs/lf_gen_faster_rcnn_x101.yml'],
		help="Path to a config file listing reader, model and optimization "
		     "parameters.",
		)
parser.add_argument(
		"--split",
		default="test",
		choices=["val", "test"],
		help="Which split to evaluate upon.",
		)
parser.add_argument(
		"--val-json",
		default="/home/ubuntu/datasets/visdial/data/visdial_1.0_val.json",
		help="Path to VisDial v1.0 val data. This argument doesn't work when "
		     "--split=test.",
		)
parser.add_argument(
		"--val-dense-json",
		default="/home/ubuntu/datasets/visdial/data/visdial_1.0_val_dense_annotations.json",
		help="Path to VisDial v1.0 val dense annotations (if evaluating on val "
		     "split). This argument doesn't work when --split=test.",
		)
parser.add_argument(
		"--test-json",
		default="/home/ubuntu/datasets/visdial/data/visdial_1.0_test.json",
		help="Path to VisDial v1.0 test data. This argument doesn't work when "
		     "--split=val.",
		)

parser.add_argument_group("Evaluation related arguments")
parser.add_argument(
		"--load-pthpaths",
		nargs="+",
		default=['/home/ubuntu/datasets/visdial/checkpoints/baselines/lf_disc_faster_rcnn_x101_trainval.pth',
		         '/home/ubuntu/datasets/visdial/checkpoints/lf_gen_bilstm/may10/checkpoint_9.pth'],
		help="Path to .pth file of pretrained checkpoint.",
		)

parser.add_argument_group(
		"Arguments independent of experiment reproducibility"
		)
parser.add_argument(
		"--gpu-ids",
		nargs="+",
		type=int,
		default=-1,
		help="List of ids of GPUs to use.",
		)
parser.add_argument(
		"--cpu-workers",
		type=int,
		default=4,
		help="Number of CPU workers for reading data.",
		)
parser.add_argument(
		"--overfit",
		action="store_true",
		help="Overfit model on 5 examples, meant for debugging.",
		)
parser.add_argument(
		"--in-memory",
		action="store_true",
		help="Load the whole dataset and pre-extracted image features in memory. "
		     "Use only in presence of large RAM, atleast few tens of GBs.",
		)

parser.add_argument_group("Submission related arguments")
parser.add_argument(
		"--save-ranks-path",
		default="logs/test_ranks.json",
		help="Path (json) to save ranks, in a EvalAI submission format.",
		)

# For reproducibility.
# Refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# =============================================================================
#   INPUT ARGUMENTS AND CONFIG
# =============================================================================

args = parser.parse_args()

# keys: {"dataset", "model", "solver"}
config = yaml.load(open(args.config_ymls[0]))

if isinstance(args.gpu_ids, int):
	args.gpu_ids = [args.gpu_ids]

device = torch.device('cuda')

# Print config and args.
print(yaml.dump(config, default_flow_style=False))

for arg in vars(args):
	print("{:<20}: {}".format(arg, getattr(args, arg)))

# =============================================================================
#   SETUP DATASET, DATALOADER, MODEL
# =============================================================================

models = []
dataloaders = []
for i, config_yml in enumerate(args.config_ymls):
	print('config_yml', config_yml)
	config = yaml.load(open(config_yml))

	if args.split == "val":
		val_dataset = VisDialDataset(
				config["dataset"],
				args.val_json,
				args.val_dense_json,
				overfit=args.overfit,
				in_memory=args.in_memory,
				return_options=True,
				add_boundary_toks=False
				if config["model"]["decoder"] == "disc"
				else True,
				)
	else:
		val_dataset = VisDialDataset(
				config["dataset"],
				args.test_json,
				overfit=args.overfit,
				in_memory=args.in_memory,
				return_options=True,
				add_boundary_toks=False
				if config["model"]["decoder"] == "disc"
				else True,
				)
	val_dataloader = DataLoader(
			val_dataset,
			batch_size=2,
			num_workers=args.cpu_workers,
			)
	dataloaders.append(iter(val_dataloader))

	encoder = Encoder(config["model"], val_dataset.vocabulary)
	decoder = Decoder(config["model"], val_dataset.vocabulary)
	print("Encoder: {}".format(config["model"]["encoder"]))
	print("Decoder: {}".format(config["model"]["decoder"]))

	# Share word embedding between encoder and decoder.
	decoder.word_embed = encoder.word_embed

	is_bilstm = config['model']['is_bilstm']

	# Wrap encoder and decoder in a model.
	model = EncoderDecoderModel(encoder, decoder, is_bilstm=is_bilstm)
	model = model.to(device)

	if args.load_pthpaths[i] != '':
		model_state_dict, _ = load_checkpoint(args.load_pthpaths[i])
		if isinstance(model, nn.DataParallel):
			model.module.load_state_dict(model_state_dict)
		else:
			model.load_state_dict(model_state_dict)
		print("Loaded model from {}".format(args.load_pthpaths[i]))
	models.append(model)

# Declare metric accumulators (won't be used if --split=test)
sparse_metrics_a = SparseGTMetrics()
ndcg_a = NDCG()

sparse_metrics_b = SparseGTMetrics()
ndcg_b = NDCG()

sparse_metrics_c = SparseGTMetrics()
ndcg_c = NDCG()

# =============================================================================
#   EVALUATION LOOP
# =============================================================================
for model in models:
	model.eval()

ranks_json = []

all_outputs = []
all_img_ids = []
all_round_ids = []

for i, batch in enumerate(tqdm(dataloaders[0])):

	for key in batch:
		batch[key] = batch[key].to(device)

	with torch.no_grad():
		output1 = models[0](batch)
		# TODO: I add this line
		output1 = torch.softmax(output1, dim=-1)

	another_batch = next(dataloaders[1])
	for key in another_batch:
		another_batch[key] = another_batch[key].to(device)

	with torch.no_grad():
		output2 = models[1](another_batch)
		# TODO: I add this line
		output2 = torch.softmax(output2, dim=-1)

	output_a = (output1 + output2) / 2.0

	disc_max_idx = output1.argmax(dim=-1)
	output2[:, :, disc_max_idx] = output1[:, :, disc_max_idx]

	output_b = output2
	output_c = (output1 + output2) / 2.0

	ranks = scores_to_ranks(output_a)
	all_outputs.append(output_a.cpu())
	all_img_ids.append(batch['img_ids'].cpu())
	all_round_ids.append(batch['num_rounds'].cpu())

	for i in range(len(batch["img_ids"])):
		# Cast into types explicitly to ensure no errors in schema.
		# Round ids are 1-10, not 0-9
		if args.split == "test":
			ranks_json.append(
					{
						"image_id": batch["img_ids"][i].item(),
						"round_id": int(batch["num_rounds"][i].item()),
						"ranks"   : [
							rank.item()
							for rank in ranks[i][batch["num_rounds"][i] - 1]
							],
						}
					)
		else:
			for j in range(batch["num_rounds"][i]):
				ranks_json.append(
						{
							"image_id": batch["img_ids"][i].item(),
							"round_id": int(j + 1),
							"ranks"   : [rank.item() for rank in ranks[i][j]],
							}
						)

	if args.split == "val":
		sparse_metrics_a.observe(output_a, batch["ans_ind"])
		sparse_metrics_b.observe(output_b, batch["ans_ind"])
		sparse_metrics_c.observe(output_c, batch["ans_ind"])

		if "gt_relevance" in batch:
			output_a = output_a[torch.arange(output_a.size(0)), batch["round_id"] - 1, :]
			ndcg_a.observe(output_a, batch["gt_relevance"])

			output_b = output_b[torch.arange(output_b.size(0)), batch["round_id"] - 1, :]
			ndcg_b.observe(output_b, batch["gt_relevance"])

			output_c = output_c[torch.arange(output_c.size(0)), batch["round_id"] - 1, :]
			ndcg_c.observe(output_c, batch["gt_relevance"])

if args.split == "val":
	all_metrics = {}
	all_metrics.update(sparse_metrics_a.retrieve(reset=True))
	all_metrics.update(ndcg_a.retrieve(reset=True))
	for metric_name, metric_value in all_metrics.items():
		print(f"{metric_name}: {metric_value}")

	all_metrics = {}
	all_metrics.update(sparse_metrics_b.retrieve(reset=True))
	all_metrics.update(ndcg_b.retrieve(reset=True))
	for metric_name, metric_value in all_metrics.items():
		print(f"{metric_name}: {metric_value}")

	all_metrics = {}
	all_metrics.update(sparse_metrics_c.retrieve(reset=True))
	all_metrics.update(ndcg_c.retrieve(reset=True))
	for metric_name, metric_value in all_metrics.items():
		print(f"{metric_name}: {metric_value}")

print("Writing ranks to {}".format(args.save_ranks_path))
os.makedirs(os.path.dirname(args.save_ranks_path), exist_ok=True)
json.dump(ranks_json, open(args.save_ranks_path, "w"))

output_path = ''.join(args.save_ranks_path.split('.')[:-1]) + '.pkl'
with open(output_path, 'wb') as f:
	pickle.dump([all_outputs, all_img_ids, all_round_ids], f)


