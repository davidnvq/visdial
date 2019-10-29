import json
import yaml
import pickle
import argparse

from tqdm import tqdm
import os
import sys
import torch
import random
import logging
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from visdial.model import get_model
from visdial.data.dataset import VisDialDataset
from visdial.metrics import SparseGTMetrics, NDCG, scores_to_ranks
from visdial.utils.checkpointing import CheckpointManager, load_checkpoint_from_config
from visdial.utils import move_to_cuda


# from options import get_comet_experiment, get_training_config_and_args

def get_num_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def evaluate(cpath, wpath, split='val', device='cuda', decoder_type='disc', ckpt_name='no_ft_ckpt_11', batch_size=1):
    sparse_metrics = SparseGTMetrics()
    ndcg = NDCG()

    with open(cpath) as file:
        config = json.load(file)

    if split == 'test':
        config['model']['test_mode'] = True
        config['dataset']['test_feat_img_path'] = config['dataset']['train_feat_img_path'].replace(
            "trainval_resnet101_faster_rcnn_genome__num_boxes",
            "test2018_resnet101_faster_rcnn_genome__num_boxes"
        )
        config['dataset']['test_json_dialog_path'] = config['dataset']['train_json_dialog_path'].replace(
            'visdial_1.0_train.json',
            'visdial_1.0_test.json'
        )

    model = get_model(config)
    if decoder_type == 'disc':
        model.decoder.gen_decoder = None

    model = model.to(device)

    print("loading weights...")
    model.load_state_dict(torch.load(wpath)['model'])

    dataset = VisDialDataset(config, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    model.eval()
    ranks_json = []

    all_outputs = []
    all_img_ids = []
    all_round_ids = []

    for idx, batch in enumerate(tqdm(dataloader)):
        torch.cuda.empty_cache()
        for key in batch:
            batch[key] = batch[key].to(device)

        with torch.no_grad():
            output = model(batch)

        for key in output:
            output[key] = torch.softmax(output[key], dim=-1)
            output = output['opt_scores']

        ranks = scores_to_ranks(output)

        all_outputs.append(output.cpu())
        all_img_ids.append(batch['img_ids'].cpu())
        all_round_ids.append(batch['num_rounds'].cpu())

        for i in range(len(batch["img_ids"])):
            if split == split:
                ranks_json.append(
                    {
                        "image_id": batch["img_ids"][i].item(),
                        "round_id": int(batch["num_rounds"][i].item()),
                        "ranks": [
                            rank.item()
                            for rank in ranks[i][0]  # [batch["num_rounds"][i] - 1]
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

        if split == 'val':
            sparse_metrics.observe(output, batch['ans_ind'])
            output = output[torch.arange(output.size(0)), batch['round_id'] - 1, :]
            ndcg.observe(output, batch["gt_relevance"])

    jpath = os.path.join(os.path.dirname(wpath), 'ranks', split, f'{ckpt_name}', f'{decoder_type}.json')
    ppath = os.path.join(os.path.dirname(wpath), 'ranks', split, f'{ckpt_name}', f'{decoder_type}.pkl')

    print("Writing ranks to {}".format(jpath))
    os.makedirs(os.path.dirname(jpath), exist_ok=True)
    json.dump(ranks_json, open(jpath, "w"))

    os.makedirs(os.path.dirname(ppath), exist_ok=True)
    with open(ppath, 'wb') as f:
        pickle.dump([all_outputs, all_img_ids, all_round_ids], f)

    all_metrics = {}
    all_metrics.update(sparse_metrics.retrieve(reset=True))
    all_metrics.update(ndcg.retrieve(reset=True))
    for metric_name, metric_value in all_metrics.items():
        print(f"{metric_name}: {metric_value}")

    return all_metrics
