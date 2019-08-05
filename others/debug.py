import torch
import torch.nn as nn

from configs.lf_disc_lstm_config import get_lf_disc_lstm_config
from visdial.data.dataset import VisDialDataset
from torch.utils.data import DataLoader

config = get_lf_disc_lstm_config()
dataset = VisDialDataset(config, split='val')
dataloader = DataLoader(dataset, batch_size=2)

for batch in dataloader:
    break

def untokenize(ids):
    "ids: numpy array"
    tokens = " ".join(dataset.tokenizer.convert_ids_to_tokens(ids))
    tokens = tokens.replace('<PAD>', ' ')
    return tokens

for key in batch:
    print('{:15}'.format(key), batch[key].size())