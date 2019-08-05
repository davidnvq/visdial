import torch
import torch.nn as nn

from visdial.common.utils import clones
from visdial.common.transformer.layer_norm import LayerNorm
from visdial.common.transformer.ffn_layer import PositionwiseFeedForward
from visdial.common.transformer.multi_attention import MultiHeadAttention

class SublayerConnection(nn.Module):

	def __init__(self, hidden_size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(hidden_size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):

	def __init__(self, hidden_size, self_attn, ffn_layer, dropout):
		super(EncoderLayer, self).__init__()
		self.size = hidden_size

		self.self_attn = self_attn
		self.ffn_layer = ffn_layer
		self.sublayers = clones(SublayerConnection(hidden_size, dropout), 2)

	def forward(self, x, mask):
		# do multi-head attention
		x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))

		# do feed-forward linear
		x = self.sublayers[1](x, self.ffn_layer)
		return x


class TransformerEncoder(nn.Module):

	def __init__(self, layer, num_layers):
		super(TransformerEncoder, self).__init__()
		self.layers = clones(layer, num_layers)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, mask):
		"""
		:param x:       [batch_size, M, hidden_size]
		:param mask:    [batch_size, M]
		:return:        [batch_size, M, hidden_size]
		"""
		for layer in self.layers:
			x = layer(x, mask)

		return x


def get_transformer_encoder(hidden_size, num_heads, d_ff, dropout, num_self_attns, **kwargs):

	self_attn = MultiHeadAttention(hidden_size, num_heads, dropout)
	ffn_layer = PositionwiseFeedForward(hidden_size, d_ff, dropout)
	encoder_layer = EncoderLayer(hidden_size, self_attn, ffn_layer, dropout)

	return TransformerEncoder(encoder_layer, num_self_attns)
