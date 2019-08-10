import torch
import torch.nn as nn

from torch.nn import functional as F
from visdial.encoders.text_encoder import TextEncoder, SummaryAttention
from visdial.common.utils import clones
from visdial.common.dynamic_rnn import DynamicRNN


class Encoder(nn.Module):

	def __init__(self, text_encoder, img_encoder, attn_encoder, hidden_size):
		super(Encoder, self).__init__()
		self.text_encoder = text_encoder
		self.img_encoder = img_encoder
		self.attn_encoder = attn_encoder
		self.summaries = clones(SummaryAttention(hidden_size), 2)

		self.encoder_linear = nn.Sequential(
				nn.Linear(hidden_size * 2, hidden_size),
				nn.ReLU(inplace=True),
				nn.Linear(hidden_size, hidden_size)
				)

	def forward(self, batch):
		BS, NH = batch['ques_len'].shape
		if self.img_encoder.split == 'test':
			NH = 1

		# [BS x NH, NR, HS] hist
		# [BS x NH, SQ, HS] ques
		# [BS x NH, NR] hist_mask
		# [BS x NR, SQ] ques_mask
		hist, ques, hist_mask, ques_mask = self.text_encoder(batch)

		# [BS x NH, NP, HS] img
		# [BS x NH, NP] img_mask
		img, img_mask = self.img_encoder(batch)

		batch_input = img, hist, ques, img_mask, hist_mask, ques_mask
		batch_input = self.attn_encoder(batch_input)

		img, hist, ques, img_mask, hist_mask, ques_mask = batch_input

		# [BS x NH, HS] img
		img = self.summaries[0](img, img_mask)

		# [BS x NH, HS] ques
		ques = self.summaries[1](ques, ques_mask)

		# [BS x NH, HS x 2]
		# In version 4, we concat [img, ques] only, there is no need for hist
		encoder_out = torch.cat([img, ques], dim=-1)

		# [BS x NH, HS]
		encoder_out = self.encoder_linear(encoder_out)

		# shape [BS, NH, HS]
		encoder_out = encoder_out.view(BS, NH, -1)
		return encoder_out


class LateFusionEncoder(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		self.word_embed = nn.Embedding(
				config['model']['vocab_size'],
				config['model']['embedding_size'],
				padding_idx=0,
				)
		self.hist_rnn = nn.LSTM(
				config['model']['embedding_size'],
				config['model']['hidden_size'],
				num_layers=2,
				batch_first=True,
				dropout=config['model']['dropout'],
				bidirectional=config['model']['bidirectional'],
				)
		self.ques_rnn = nn.LSTM(
				config['model']['embedding_size'],
				config['model']['hidden_size'],
				num_layers=2,
				batch_first=True,
				dropout=config['model']['dropout'],
				bidirectional=config['model']['bidirectional'],
				)

		self.hist_linear = nn.Linear(config['model']['hidden_size'] * 2,
		                             config['model']['hidden_size'])

		self.ques_linear = nn.Linear(config['model']['hidden_size'] * 2,
		                             config['model']['hidden_size'])

		self.dropout = nn.Dropout(p=config['model']['dropout'])

		# questions and history are right padded sequences of variable length
		# use the DynamicRNN utility module to handle them properly
		self.hist_rnn = DynamicRNN(self.hist_rnn)
		self.ques_rnn = DynamicRNN(self.ques_rnn)

		# project image features to lstm_hidden_size for computing attention
		self.image_features_projection = nn.Linear(
				config['model']["img_feature_size"], config['model']['hidden_size']
				)
		# nn.init.kaiming_uniform_(self.image_features_projection.weight)
		# nn.init.constant_(self.image_features_projection.bias, 0)

		# fc layer for image * question to attention weights
		self.attention_proj = nn.Linear(config['model']['hidden_size'], 1)
		# nn.init.kaiming_uniform_(self.attention_proj.weight)
		# nn.init.constant_(self.attention_proj.bias, 0)

		# fusion layer (attended_image_features + question + history)
		fusion_size = (
				config['model']["img_feature_size"] + config['model']['hidden_size'] * 2
		)
		self.fusion = nn.Linear(fusion_size, config['model']['hidden_size'])
		# nn.init.kaiming_uniform_(self.fusion.weight)
		# nn.init.constant_(self.fusion.bias, 0)

	def forward(self, batch, debug=False):
		# shape: (batch_size, img_feature_size) - CNN fc7 features
		# shape: (batch_size, num_proposals, img_feature_size) - RCNN features
		# shape: [BS, NP, IS]
		img = batch["img_feat"]

		# shape: [BS, 10, SEQ]
		ques = batch["ques_tokens"]

		# shape: [BS, 10, SEQ x 2 x 10] <- concatenated q & a * 10 rounds
		hist = batch["concat_hist_tokens"]

		# num_rounds = 10, even for test (padded dialog rounds at the end)
		(BS, NR, SEQ), HS = ques.size(), self.config['model']['hidden_size']
		NP, IS = img.size(1), img.size(2)

		# embed questions
		# shape: [BS x NR, SEQ]
		ques = ques.view(BS * NR, SEQ)

		# shape: [BS x NR, SEQ, WE]
		ques_embed = self.word_embed(ques)

		# shape: [num_layers x 1, BS x NR, HS] in 1 directional
		ques_len = batch['ques_len'].view(-1, )
		_, (ques_embed, _) = self.ques_rnn(ques_embed, ques_len)

		# embed history
		# shape: [BS x NR, SEQ x 20]
		hist = hist.view(BS * NR, SEQ * 20)

		# shape: [BS x NR, SEQ x 20, WE]
		hist_embed = self.word_embed(hist)

		# shape: [num_layers x 1, BS x NR, HS] in 1 directional
		# shape: [num_layers x 2, BS x NR, HS] in 2 directionals
		hist_len = batch['concat_hist_len'].view(-1)

		_, (hist_embed, _) = self.hist_rnn(hist_embed, hist_len)

		# shape: [2, BS x NR, HS] <- select the last layer
		ques_embed = ques_embed[-2:]
		# shape: [BS x NR, HS x 2]
		ques_embed = torch.cat([ques_embed[0], ques_embed[1]], dim=-1)
		# shape: [BS x NR, HS]
		ques_embed = self.ques_linear(ques_embed)

		# shape: [2, BS x NR, HS]
		hist_embed = hist_embed[-2:]
		# shape: [BS x NR, HS x 2]
		hist_embed = torch.cat([hist_embed[0], hist_embed[1]], dim=-1)
		# shape: [BS x NR, HS]
		hist_embed = self.hist_linear(hist_embed)

		# project down image features and ready for attention
		# shape: [BS, NP, HS]
		projected_image_features = self.image_features_projection(img)

		# TODONE: below lines are the same as baseline

		# shape: [BS, 1, NP, HS]
		projected_image_features = projected_image_features.view(BS, 1, -1, HS)

		# shape: [BS, NR, 1, HS]
		projected_ques_features = ques_embed.view(BS, NR, 1, HS)

		# shape: [BS, NR, NP, HS]
		projected_ques_image = projected_image_features * projected_ques_features
		projected_ques_image = self.dropout(projected_ques_image)

		# computing attention weights
		# shape: [BS, NR, NP, 1]
		image_attention_weights = self.attention_proj(projected_ques_image)

		# shape: [BS, NR, NP, 1]
		image_attention_weights = F.softmax(image_attention_weights, dim=-2)  # <- dim = NP

		# shape: [BS, 1, NP, IS]
		img = img.view(BS, 1, NP, IS)

		# shape: [BS, NR, NP, 1] * [BS, (1), NP, IS] -> [BS, NR, NP, IS]
		attended_image_features = image_attention_weights * img

		# shape: [BS, NR, IS]
		img = attended_image_features.sum(dim=-2)  # dim=NP

		# shape: [BS x NR, IS]
		img = img.view(BS * NR, IS)

		# shape: [BS x NR, IS + HSx2]
		fused_vector = torch.cat((img, ques_embed, hist_embed), 1)
		fused_vector = self.dropout(fused_vector)

		# shape: [BS x NR, HS]
		fused_embedding = torch.tanh(self.fusion(fused_vector))

		# shape: [BS, NR, HS]
		fused_embedding = fused_embedding.view(BS, NR, -1)

		if debug:
			return fused_embedding, image_attention_weights.squeeze(-1)
		return fused_embedding
