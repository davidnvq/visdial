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
from visdial.utils import move_to_cuda
from visdial.metrics import SparseGTMetrics, NDCG, scores_to_ranks

CKPT1='/content/gdrive/My Drive/checkpoints/lf_disc/may13/checkpoint_best_ndcg.pth'
CKPT2='/content/gdrive/My Drive/checkpoints/lf_disc/may13/checkpoint_best_mean.pth'
CKPT3='/content/gdrive/My Drive/checkpoints/lf_gen/checkpoint_9.pth'

CONFIG1='/content/visdial/configs/lf_disc_faster_rcnn_x101.yml'
CONFIG2='/content/visdial/configs/lf_disc_faster_rcnn_x101.yml'
CONFIG3='/content/visdial/configs/lf_gen_faster_rcnn_x101.yml'

parser = argparse.ArgumentParser("Evaluate and/or generate EvalAI submission file.")
parser.add_argument('--config-ymls', nargs='+', default=[CONFIG1,CONFIG2, CONFIG3 ])
parser.add_argument("--image-features-h5", default="")
parser.add_argument("--split", default="val")
parser.add_argument("--json-word-counts", default="")
parser.add_argument('--json-dialogs', default='')
parser.add_argument('--device', default='cuda')
parser.add_argument('--json-dense', default='')
parser.add_argument('--load-pthpaths', nargs='+', default=[CKPT1, CKPT2, CKPT3])
parser.add_argument('--gpu-ids', default=[0], nargs='+', type=int)
parser.add_argument('--cpu-workers', default=4, type=int)
parser.add_argument('--overfit', action='store_true')
parser.add_argument("--save-ranks-path", default='/content/gdrive/My Drive/checkpoints/ensemble.json')

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

if isinstance(args.gpu_ids, int):
	args.gpu_ids = [args.gpu_ids]

device = 'cuda'

# =============================================================================
#   SETUP DATASET, DATALOADER, MODEL
# =============================================================================
models = []
dataloaders = []
for i, config_yml in enumerate(args.config_ymls):
	config = yaml.load(open(config_yml))
	is_abtoks = True if config["model"]["decoder"] == "gen" else False
	dataset = VisDialDataset(
			config["dataset"],
			jsonpath_dialogs=args.json_dialogs,
			jsonpath_vocab=args.json_word_counts,
			jsonpath_dense=args.json_dense,
			hdfpath_img_features=args.image_features_h5,
			overfit=args.overfit,
			return_options=True,
			add_boundary_toks=is_abtoks,
			)

	dataloader = DataLoader(
			dataset,
			batch_size=2,
			num_workers=args.cpu_workers,
			)
	dataloaders.append(iter(dataloader))

	encoder = Encoder(config["model"], dataset.vocabulary)
	decoder = Decoder(config["model"], dataset.vocabulary)
	print("Encoder: {}".format(config["model"]["encoder"]))
	print("Decoder: {}".format(config["model"]["decoder"]))

	# Share word embedding between encoder and decoder.
	decoder.word_embed = encoder.word_embed

	# Wrap encoder and decoder in a model.
	model = EncoderDecoderModel(encoder, decoder)
	model = model.to(device)

	model = load_checkpoint(args.load_pthpaths[i], model, device=device)
	models.append(model)

# Declare metric accumulators (won't be used if --split=test)
sparse_metrics_a = SparseGTMetrics()
ndcg_a = NDCG()

# =============================================================================
#   EVALUATION LOOP
# =============================================================================
for model in models:
	model.eval()

ranks_json = []
all_outputs = []
all_img_ids = []
all_round_ids = []

for i in range(len(dataloaders[0])):
	print('batch index', i)

	for j, net in enumerate(models):
		with torch.no_grad():
			batch = next(dataloaders[j])
			batch = move_to_cuda(batch, device)
			output = net(batch)
			output = torch.softmax(output, dim=-1)
			if j == 0:
				cu_output = output
			else:
				cu_output += output

	cu_output /= float(len(dataloaders)) # cummulative output

	ranks = scores_to_ranks(cu_output)

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
		sparse_metrics_a.observe(cu_output, batch["ans_ind"])

		if "gt_relevance" in batch:
			cu_output = cu_output[torch.arange(cu_output.size(0)), batch["round_id"] - 1, :]
			ndcg_a.observe(cu_output, batch["gt_relevance"])

	torch.cuda.empty_cache()


if args.split == "val":
	all_metrics = {}
	all_metrics.update(sparse_metrics_a.retrieve(reset=True))
	all_metrics.update(ndcg_a.retrieve(reset=True))
	for metric_name, metric_value in all_metrics.items():
		print(f"{metric_name}: {metric_value}")

print("Writing ranks to {}".format(args.save_ranks_path))
os.makedirs(os.path.dirname(args.save_ranks_path), exist_ok=True)
json.dump(ranks_json, open(args.save_ranks_path, "w"))

