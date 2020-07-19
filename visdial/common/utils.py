import copy
import torch.nn as nn


def clones(module, N):
    "Produce N identical modules"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def check_flag(d, key):
    "Check whether the dictionary `d` has `key` and `d[key]` is True"
    return d.get(key) is not None and d.get(key)
