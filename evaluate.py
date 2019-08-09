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
from visdial.model import VisdialModel
from visdial.data.dataset import VisDialDataset
from visdial.utils.checkpointing import load_checkpoint
from visdial.utils import move_to_cuda
from visdial.metrics import SparseGTMetrics, NDCG, scores_to_ranks
from visdial.model import get_model


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="attn_misc_lstm")
parser.add_argument("--weights", default="")
parser.add_argument("--split", default="")
parser.add_argument("--decoder_type", default='misc')
parser.add_argument("--save-ranks-path", default="logs/val_{}_ranks.json")

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

from configs import get_config

config = get_config(config_name=args.config)
print(json.dumps(config, indent=2))

# =============================================================================
#   SETUP DATASET, DATALOADER, MODEL
# =============================================================================
dataset = VisDialDataset(config, split=args.split)
dataloader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=config['solver']['cpu_workers'])

device = 'cuda'
model = get_model(config)
model = model.to(device)
print("Loading weights...")
model.load_state_dict(torch.load(args.weights)['model'])
print("Finish loading weights...")
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

	for decoder_type in output:
		output[decoder_type] = torch.softmax(output[decoder_type], dim=-1)

	if args.decoder_type == 'disc':
		output = output['opt_scores']
	elif args.decoder_type == 'gen':
		output = output['opts_out_scores']
	elif args.decoder_type == 'misc':
		output = (output['opt_scores'] + output['opts_out_scores'])/2.0

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
save_path = args.save_ranks_path.format(args.decoder_type)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
json.dump(ranks_json, open(save_path, "w"))

# output_path = ''.join(args.save_ranks_path.split('.')[:-1]) + '.pkl'
# with open(output_path, 'wb') as f:
# 	pickle.dump([all_outputs, all_img_ids, all_round_ids], f)
