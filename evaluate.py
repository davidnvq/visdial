import os
import json
import yaml
import pickle
import argparse
import nltk
nltk.download('punkt')

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


parser = argparse.ArgumentParser()
parser.add_argument("--image-features-h5", default="")
parser.add_argument("--split", default="val")
parser.add_argument("--json-word-counts", default="")
parser.add_argument("--device", default="cpu")
parser.add_argument("--batch-size", default=4, type=int)
parser.add_argument("--overfit", action="store_true")
parser.add_argument("--gpu-ids", nargs="+", default=0, type=int)
parser.add_argument("--cpu-workers", default=4, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--config-yml", default="")
parser.add_argument("--json-dialogs", default="")
parser.add_argument("--json-dense", default="")
parser.add_argument("--load-pthpath", default="")
parser.add_argument("--save-ranks-path", default="logs/ranks.json")

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
config = yaml.load(open(args.config_yml))

if isinstance(args.gpu_ids, int):
	args.gpu_ids = [args.gpu_ids]

device = args.device

# Print config and args.
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
	print("{:<20}: {}".format(arg, getattr(args, arg)))

# =============================================================================
#   SETUP DATASET, DATALOADER, MODEL
# =============================================================================
is_abtoks = True if config["model"]["decoder"] != "disc" else False
is_return = True if config["model"]["decoder"] == "disc" else False

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
		batch_size=args.batch_size,
		num_workers=args.cpu_workers
		)

# Pass vocabulary to construct Embedding layer.
encoder = Encoder(config["model"], dataset.vocabulary)
decoder = Decoder(config["model"], dataset.vocabulary)

print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))

# Share word embedding between encoder and decoder.
decoder.word_embed = encoder.word_embed

# Wrap encoder and decoder in a model.
model = EncoderDecoderModel(encoder, decoder)
model = model.to(device)
model = load_checkpoint(args.load_pthpath, model, device=args.device)

if len(args.gpu_ids) > 1:
	model = nn.DataParallel(model, args.gpu_ids)

# Declare metric accumulators (won't be used if --split=test)
sparse_metrics = SparseGTMetrics()
ndcg = NDCG()

# =============================================================================
#   EVALUATION LOOP
# =============================================================================

model.eval()
ranks_json = []

all_outputs = []
all_img_ids = []
all_round_ids = []

num_imgs = 0

for _, batch in enumerate(tqdm(dataloader)):
	num_imgs += len(batch['img_ids'])

	for key in batch:
		batch[key] = batch[key].to(device)

	with torch.no_grad():
		output = model(batch)

	# TODO: I add this line
	output = torch.softmax(output, dim=-1)

	ranks = scores_to_ranks(output)
	all_outputs.append(output.cpu())
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
		sparse_metrics.observe(output, batch["ans_ind"])
		if "gt_relevance" in batch:
			output = output[
			         torch.arange(output.size(0)), batch["round_id"] - 1, :
			         ]
			ndcg.observe(output, batch["gt_relevance"])

print('num_ings', num_imgs)

if args.split == "val":
	all_metrics = {}
	all_metrics.update(sparse_metrics.retrieve(reset=True))
	all_metrics.update(ndcg.retrieve(reset=True))
	for metric_name, metric_value in all_metrics.items():
		print(f"{metric_name}: {metric_value}")

print("Writing ranks to {}".format(args.save_ranks_path))
os.makedirs(os.path.dirname(args.save_ranks_path), exist_ok=True)
json.dump(ranks_json, open(args.save_ranks_path, "w"))

output_path = ''.join(args.save_ranks_path.split('.')[:-1]) + '.pkl'
with open(output_path, 'wb') as f:
	pickle.dump([all_outputs, all_img_ids, all_round_ids], f)
