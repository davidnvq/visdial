import torch
import torch.nn as nn
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

	def __init__(self, hidden_size, num_heads, memory_size=1, dropout=0.0):
		super().__init__()

		self.hidden_size = hidden_size
		self.memory_size = memory_size
		self.num_heads = num_heads
		self.d_h = hidden_size // num_heads
		self.dropout = nn.Dropout(p=dropout)

		self.x_proj_linear = nn.Linear(hidden_size, hidden_size, bias=False)
		self.y_proj_linear = nn.Linear(hidden_size, hidden_size, bias=False)
		# nn.init.kaiming_uniform_(self.x_proj_linear.weight)
		# nn.init.kaiming_uniform_(self.y_proj_linear.weight)

		self.x_memory = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(memory_size, hidden_size)))
		self.y_memory = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(memory_size, hidden_size)))

		self.attn_X_guided_by_Y = None
		self.attn_Y_guided_by_X = None

	def project(self, x, x_mem, linear):
		x_mem_size = x.size(0), self.memory_size, self.hidden_size
		x = torch.cat([x_mem.unsqueeze(0).expand(*x_mem_size), x], dim=1)
		x_proj = linear(x)
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
	def __init__(self, hidden_size, num_heads, memory_size=1, dropout=0.0):
		super(CrossAttentionLayer, self).__init__()
		self.attns = clones(MultiHeadAttention(hidden_size, num_heads, memory_size, dropout), 3)
		self.norms = clones(NormalSubLayer(hidden_size, dropout), 3)


	def forward(self, triples):
		"""
		:param x:       [batch_size, M, hidden_size]
		:param mask_x:  [batch_size, M]
		:return:        [batch_size, M, hidden_size]
		"""
		x, y, z, mask_x, mask_y, mask_z = triples
		x_in_y, y_in_x = self.attns[0](x, y, mask_x, mask_y)
		x_in_z, z_in_x = self.attns[1](x, z, mask_x, mask_z)
		y_in_z, z_in_y = self.attns[2](y, z, mask_y, mask_z)

		x = self.norms[0](torch.cat([x, y_in_x, z_in_x], dim=-1))
		z = self.norms[1](torch.cat([z, y_in_z, x_in_z], dim=-1))
		y = self.norms[2](torch.cat([y, z_in_y, x_in_y], dim=-1))
		return x, y, z, mask_x, mask_y, mask_z


class CrossAttentionEncoder(nn.Module):

	def __init__(self, hidden_size, num_heads, share_attn, memory_size=1, dropout=0.0, num_cross_attns=2, **kwargs):
		super(CrossAttentionEncoder, self).__init__()

		# share the same cross-attn layer
		# then the number of params reduced by / num_cross_attns
		# check very careful
		if share_attn:
			layers = [CrossAttentionLayer(hidden_size, num_heads, memory_size, dropout)] * num_cross_attns
		else:
			layers = [CrossAttentionLayer(hidden_size, num_heads, memory_size, dropout) for i in range(num_cross_attns)]

		self.cross_attn_encoder = nn.Sequential(*layers)

	def forward(self, triples):
		return self.cross_attn_encoder(triples)
