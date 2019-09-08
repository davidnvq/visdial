import torch
import torch.nn as nn

import logging

try:
	from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
except (ImportError, AttributeError) as e:
	logging.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
	LayerNorm = torch.nn.LayerNorm

from visdial.common.utils import clones


class NormalSubLayer(nn.Module):

	def __init__(self, hidden_size, dropout):
		super(NormalSubLayer, self).__init__()
		self.linear = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size),
		                            nn.ReLU(inplace=True),
		                            nn.Dropout(p=dropout))

	def forward(self, x):
		"""x: shape [batch_size, M, hidden_size*3]"""
		return self.linear(x)


class MultiHeadAttention(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config

		self.hidden_size = config['model']['hidden_size']
		self.num_heads = config['model']['ca_num_cross_attn_heads']
		self.memory_size = config['model']['ca_memory_size']
		self.dropout = nn.Dropout(p=config['model']['dropout'])

		self.d_h = self.hidden_size // self.num_heads

		if self.config['model']['ca_has_proj_linear']:
			self.x_proj_linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
			self.y_proj_linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
		else:
			self.x_proj_linear = None
			self.y_proj_linear = None

		self.x_memory = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.memory_size, self.hidden_size)))
		self.y_memory = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.memory_size, self.hidden_size)))

		self.attn_X_guided_by_Y = None
		self.attn_Y_guided_by_X = None

	def project(self, x, x_mem, linear=None):
		x_mem_size = x.size(0), self.memory_size, self.hidden_size
		x = torch.cat([x_mem.unsqueeze(0).expand(*x_mem_size), x], dim=1)

		if self.config['model']['ca_has_proj_linear']:
			x_proj = linear(x)
			x_proj = x_proj.view(x_proj.size(0), x_proj.size(1), self.num_heads, self.d_h)
		else:
			x_proj = x.view(x.size(0), x.size(1), self.num_heads, self.d_h)
		return x, x_proj


	def forward(self, x, y, mask_x, mask_y):
		"""
		x: shape: [batch_size, M, hidden_size]
		y: shape: [batch_size, N, hidden_size]
		mask_x: shape: [batch_size, M]
		mask_y: shape: [batch_size, N]
		"""
		memory_mask = x.new_ones((x.size(0), self.memory_size)).long()

		mask_x = torch.cat([memory_mask, mask_x], dim=1)
		mask_y = torch.cat([memory_mask, mask_y], dim=1)
		M_mem, N_mem = mask_x.size(1), mask_y.size(1)

		mask_x = mask_x[:, None, :, None].repeat(1, self.num_heads, 1, N_mem)
		mask_y = mask_y[:, None, None, :].repeat(1, self.num_heads, M_mem, 1)

		X_mem, X_proj = self.project(x, self.x_memory, self.x_proj_linear)
		Y_mem, Y_proj = self.project(y, self.y_memory, self.y_proj_linear)

		# (1) shape [bs, num_heads, mem_size + M, d_h]
		# (2) shape [bs, num_heads, d_h, mem_size + N]
		X_proj = X_proj.permute(0, 2, 1, 3)
		Y_proj = Y_proj.permute(0, 2, 3, 1)

		# shape: [bs, num_heads, mem_size + M, mem_size + N]
		affinity_matrix = torch.matmul(X_proj, Y_proj)
		affinity_matrix = affinity_matrix.masked_fill(mask_x == 0, -1e9)
		affinity_matrix = affinity_matrix.masked_fill(mask_y == 0, -1e9)

		attn_X_guided_by_Y = torch.softmax(affinity_matrix, dim=2)
		attn_Y_guided_by_X = torch.softmax(affinity_matrix, dim=3)

		# (1) shape [bs, mem_size + M, mem_size + N]
		# (2) shape [bs, mem_size + M, mem_size + N]
		attn_X_guided_by_Y = torch.mean(attn_X_guided_by_Y, dim=1)
		attn_Y_guided_by_X = torch.mean(attn_Y_guided_by_X, dim=1)

		# self.attn_X_guided_by_Y = attn_X_guided_by_Y
		# self.attn_Y_guided_by_X = attn_Y_guided_by_X

		# (1) shape: [bs, mem_size + N, hidden_size]
		# (2) shape: [bs, mem_size + M, hidden_size]
		X_attends_in_Y = torch.matmul(attn_X_guided_by_Y.transpose(1, 2), X_mem)
		Y_attends_in_X = torch.matmul(attn_Y_guided_by_X, Y_mem)

		X_attends_in_Y = X_attends_in_Y[:, self.memory_size:, :]
		Y_attends_in_X = Y_attends_in_X[:, self.memory_size:, :]
		return X_attends_in_Y, Y_attends_in_X


class CrossAttentionLayer(nn.Module):
	def __init__(self, config):
		super(CrossAttentionLayer, self).__init__()
		self.config = config
		hidden_size = config['model']['hidden_size']
		dropout = config['model']['dropout']

		self.attns = clones(MultiHeadAttention(config), 3)
		self.im_mlp = NormalSubLayer(hidden_size, dropout)
		self.qe_mlp = NormalSubLayer(hidden_size, dropout)

		if self.config['model']['ca_has_updated_hist']:
			self.hi_mlp = NormalSubLayer(hidden_size, dropout)

		if self.config['model']['ca_has_layer_norm']:
			self.im_norm = LayerNorm(hidden_size)
			self.qe_norm = LayerNorm(hidden_size)
			if self.config['model']['ca_has_updated_hist']:
				self.hi_norm = LayerNorm(hidden_size)

	def forward(self, triples):
		"""
		:param triples: (im, qe, hi, mask_i, mask_qe, mask_hi)
		:return: (im, qe, hi, mask_i, mask_qe, mask_hi)
		im [batch_size, M, hidden_size]
		"""
		im, qe, hi, mask_im, mask_qe, mask_hi = triples
		im_in_qe, qe_in_im = self.attns[0](im, qe, mask_im, mask_qe)
		im_in_hi, hi_in_im = self.attns[1](im, hi, mask_im, mask_hi)
		qe_in_hi, hi_in_qe = self.attns[2](qe, hi, mask_qe, mask_hi)

		a_im = self.im_mlp(torch.cat([im, qe_in_im, hi_in_im], dim=-1))
		a_qe = self.qe_mlp(torch.cat([qe, hi_in_qe, im_in_qe], dim=-1))

		# The best one doesn't need this
		if self.config['model']['ca_has_updated_hist']:
			a_hi = self.hi_mlp(torch.cat([hi, qe_in_hi, im_in_hi], dim=-1))

		if self.config['model']['ca_has_residual']:
			im = im + a_im
			qe = qe + a_qe
			if self.config['model']['ca_has_updated_hist']:
				hi = hi + a_hi
		else:
			im = a_im
			qe = a_qe
			if self.config['model']['ca_has_updated_hist']:
				hi = a_hi

		if self.config['model']['ca_has_layer_norm']:
			im = self.im_norm(im)
			qe = self.qe_norm(qe)
			if self.config['model']['ca_has_updated_hist']:
				hi = self.hi_norm(hi)

		return im, qe, hi, mask_im, mask_qe, mask_hi


class CrossAttentionEncoder(nn.Module):

	def __init__(self, config):
		"""
		hidden_size, num_heads, share_attn, memory_size=1, dropout=0.0, num_cross_attns=2
		:param config:
		"""
		super(CrossAttentionEncoder, self).__init__()

		self.config = config

		num_cross_attns = self.config['model']['ca_num_cross_attns']

		if self.config['model']['ca_has_shared_attns']:
			layers = [CrossAttentionLayer(config)] * num_cross_attns
		else:
			layers = [CrossAttentionLayer(config) for _ in range(num_cross_attns)]

		self.cross_attn_encoder = nn.Sequential(*layers)

	def forward(self, triples):
		return self.cross_attn_encoder(triples)