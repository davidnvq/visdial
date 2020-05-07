import json
import argparse

import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from visdial.data.dataset import VisDialDataset
from visdial.metrics import SparseGTMetrics, NDCG, scores_to_ranks

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default='checkpoints/model_v1.pth')
parser.add_argument("--split", default="test")
parser.add_argument("--decoder_type", default='disc')
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--output_path", default="checkpoints/val.json")

args = parser.parse_args()
device = args.device
split = args.split
decoder_type = args.decoder_type
model = torch.load(args.model_path)
config = model.encoder.config

test_mode = False
if args.split == 'test':
    test_mode = True
    config['dataset']['test_feat_img_path'] = config['dataset']['train_feat_img_path'].replace(
        "trainval_resnet101_faster_rcnn_genome__num_boxes",
        "test2018_resnet101_faster_rcnn_genome__num_boxes"
    )
    config['dataset']['test_json_dialog_path'] = config['dataset']['train_json_dialog_path'].replace(
        'visdial_1.0_train.json',
        'visdial_1.0_test.json'
    )

model = model.to(device)

sparse_metrics = SparseGTMetrics()
ndcg = NDCG()

dataset = VisDialDataset(config, split=args.split)
dataloader = DataLoader(dataset, batch_size=1)

model = model.eval()
ranks_json = []

for idx, batch in enumerate(tqdm(dataloader)):
    torch.cuda.empty_cache()
    for key in batch:
        batch[key] = batch[key].to(device)

    with torch.no_grad():
        output = model(batch, test_mode=test_mode)

    if decoder_type == 'misc':
        output = (output['opts_out_scores'] + output['opt_scores']) / 2.0
    elif decoder_type == 'disc':
        output = output['opt_scores']
    elif decoder_type == 'gen':
        output = output['opts_out_scores']
    ranks = scores_to_ranks(output)

    for i in range(len(batch["img_ids"])):
        if split == split:
            ranks_json.append(
                {
                    "image_id": batch["img_ids"][i].item(),
                    "round_id": int(batch["num_rounds"][i].item()),
                    "ranks": [
                        rank.item()
                        for rank in ranks[i][0]
                    ],
                }
            )
        else:
            for j in range(batch["num_rounds"][i]):
                ranks_json.append(
                    {
                        "image_id": batch["img_ids"][i].item(),
                        "round_id": int(j + 1),
                        "ranks": [rank.item() for rank in ranks[i][j]],
                    }
                )

    if split == 'val' and not config['dataset']['v0.9']:
        sparse_metrics.observe(output, batch['ans_ind'])
        output = output[torch.arange(output.size(0)), batch['round_id'] - 1, :]
        ndcg.observe(output, batch["gt_relevance"])

jpath = args.output_path

print("Writing ranks to {}".format(jpath))
os.makedirs(os.path.dirname(jpath), exist_ok=True)
json.dump(ranks_json, open(jpath, "w"))

all_metrics = {}
all_metrics.update(sparse_metrics.retrieve(reset=True))
all_metrics.update(ndcg.retrieve(reset=True))
for metric_name, metric_value in all_metrics.items():
    print(f"{metric_name}: {metric_value}")
