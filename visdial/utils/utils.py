import copy
import torch.nn as nn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def check_flag(d, key):
    return d.get(key) is not None and d.get(key)


def get_num_params(module):
    """Compute the number of parameters of the module"""
    pp = 0
    for p in list(module.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def move_to_cuda(batch, device):
    for key in batch:
        batch[key] = batch[key].to(device)
    return batch
