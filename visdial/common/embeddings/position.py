import math
import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):

    def __init__(self, embedding_size, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embedding_size).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embedding_size, 2).float() * -(math.log(10000.0) / embedding_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: shape [bs, seq_len] expect!
        :return:  shape [bs, seq_len, hidden_size]
        """
        # shape [BS, seq_len]
        y = x.view(-1, x.size(-1)) if len(x.size()) > 2 else x
        # shape [1, seq_len, embedding_size]
        out = self.pe[:, :y.size(1)]

        # shape [BS, seq_len, embedding_size]
        out = out.repeat(y.size(0), 1, 1)

        # shape [*x.size(), embedding_size]
        return out.view(*x.size(), -1)