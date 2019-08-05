import torch
import torch.nn as nn

from .attention import Attention
from ..utils import clones

class MultiHeadAttention(nn.Module):

	def __init__(self, hidden_size, num_heads, dropout=0.1):
		super(MultiHeadAttention, self).__init__()

		self.hidden_size = hidden_size
		self.num_heads = num_heads
		self.d_h = hidden_size // num_heads

		self.dropout = nn.Dropout(p=dropout)
		self.attn_fn = Attention(dropout)

		self.linears = clones(nn.Linear(hidden_size, hidden_size), 4)
		for m in self.linears:
			nn.init.kaiming_uniform_(m.weight)
			nn.init.constant_(m.bias)

		self.attn = None


	def forward(self, query, key, value, mask=None):
		"""
		:param query:   [batch_size, M, hidden_size]
		:param key:     [batch_size, N, hidden_size]
		:param value:   [batch_size, N, hidden_size]
		:param mask:    [batch_size, N]
		:return:
		"""

		if mask is not None:
			mask = mask.unsqueeze(1)
		else:
			mask = value.new_ones(value.size()).unsqueeze(1)

		batch_size = query.size(0)

		# Do all linear projections and split hidden_size -> num_heads x d_h
		query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_h).transpose(1, 2)
		                     for l, x in zip(self.linears, (query, key, value))]

		# apply attention
		x, self.attn = self.attn_fn(query, key, value, mask)

		# concat and apply the final linear
		# shape [batch_size, M, num_heads, d_h]
		x = x.transpose(1, 2).contiguous()
		x = x.view(batch_size, -1, self.num_heads * self.d_h)
		return self.linears[-1](x)