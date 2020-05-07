import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Compute the Positional Embedding based on
    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    https://arxiv.org/pdf/1810.04805
    """

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
        Arguments
        ---------
        x: torch.FloatTensor
        	The input tensor which is a sequence of tokens
        	Shape [batch_size, seq_len, ...] is expected!
        Returns
        -------
        pos_embedding: torch.FloatTensor
            The positional embeddings for all the tokens in the sequence!
            Shape [batch_size, seq_len, hidden_size]
        """

        # shape [BS, seq_len]
        bs, seq_len = x.size(0), x.size(1)

        # shape [1, seq_len, embedding_size]
        pos_embedding = self.pe[:, :seq_len]

        # shape [BS, seq_len, embedding_size]
        pos_embedding = pos_embedding.repeat(bs, 1, 1)

        return pos_embedding
