import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """This module perform self-attention on an utility
    to summarize it into a single vector."""

    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.attn_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )
        self.attn_weights = None

    def forward(self, x, mask_x):
        """
        Arguments
        ---------
        x: torch.FloatTensor
        	The input tensor which is a sequence of tokens
        	Shape [batch_size, M, hidden_size]
        mask_x: torch.LongTensor
            The mask of the input x where 0 represents the <PAD> token
        	Shape [batch_size, M]
        Returns
        -------
        summarized_vector: torch.FloatTensor
            The summarized vector of the utility (the context vector for this utility)
            Shape [batch_size, hidden_size]
        """

        # shape [bs, M, 1]
        attn_weights = self.attn_linear(x)
        attn_weights = attn_weights.masked_fill(mask_x.unsqueeze(-1) == 0, value=-9e10)
        attn_weights = torch.softmax(attn_weights, dim=-2)
        self.attn_weights = attn_weights

        # shape [bs, 1, hidden_size]
        summarized_vector = torch.matmul(attn_weights.transpose(-2, -1), x)
        summarized_vector = summarized_vector.squeeze(dim=-2)
        return summarized_vector
