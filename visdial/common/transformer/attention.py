import math
import torch
import torch.nn as nn


class Attention(nn.Module):

	def __init__(self, dropout=0.0):
		super(Attention, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, query, key, value, mask):
		"""
		:param query:   [batch_size, num_heads, M, d_h]
		:param key:     [batch_size, num_heads, N, d_h]
		:param value:   [batch_size, num_heads, N, d_h]
		:param mask:    [batch_size, num_heads, N]
		:return:
		"""
		key = key.transpose(-2, -1)
		d_h = key.size(-1)

		# shape [batch_size, num_heads, M, N]
		mask = mask.unsqueeze(2).repeat(1, 1, query.size(2), 1)

		# shape [batch_size, num_heads, M, N]
		scores = torch.matmul(query, key) / math.sqrt(d_h)
		scores = scores.masked_fill(mask == 0, -1e9)
		p_attn = torch.softmax(scores, dim=-1)
		p_attn = self.dropout(p_attn)

		return torch.matmul(p_attn, value), p_attn



