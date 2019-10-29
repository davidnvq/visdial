import copy
import torch.nn as nn

def clones(module, N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def check_flag(d, key):
	return d.get(key) is not None and d.get(key)