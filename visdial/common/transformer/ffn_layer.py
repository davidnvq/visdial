import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):

	def __init__(self, hidden_size, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.ffn = nn.Sequential(nn.Linear(hidden_size, d_ff),
		                         nn.ReLU(inplace=True),
		                         nn.Linear(d_ff, hidden_size),
		                         nn.Dropout(p=dropout))
		nn.init.kaiming_uniform_(self.ffn[0].weight)
		nn.init.kaiming_uniform_(self.ffn[2].weight)
		nn.init.constant_(self.ffn[0].bias, 0)
		nn.init.constant_(self.ffn[2].bias, 0)


	def forward(self, x):
		return self.ffn(x)
