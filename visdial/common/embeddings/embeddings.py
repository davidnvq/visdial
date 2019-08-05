import torch.nn as nn
from .position import PositionalEmbedding

class TextEmbeddings(nn.Module):


	def __init__(self, vocab_size, embedding_size,  hidden_size, has_position=False, has_hidden_layer=False, **kwargs):
		super(TextEmbeddings, self).__init__()
		self.has_position = has_position
		self.has_hidden_layer = has_hidden_layer
		self.tok_embedding = nn.Embedding(vocab_size, embedding_size, 0)
		if has_position:
			self.pos_embedding = PositionalEmbedding(embedding_size)
		if has_hidden_layer:
			self.linear = nn.Linear(embedding_size, hidden_size)

	def forward(self, tokens):
		"""
		:param tokens:  [batch_size, num_rounds, seq_len]
		:return:        [batch_size, num_rounds, seq_len, embedding_size]
		"""
		tok_embed = self.tok_embedding(tokens)

		if self.has_position:
			pos_embed = self.pos_embedding(tokens)
			res_embed = tok_embed + pos_embed
		else:
			res_embed = tok_embed

		if self.has_hidden_layer:
			res_embed = self.linear(res_embed)

		return res_embed