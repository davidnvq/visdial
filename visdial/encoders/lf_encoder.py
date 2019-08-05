import torch
from torch import nn
from torch.nn import functional as F
from visdial.encoders.img_encoder import ImageEncoder
from visdial.encoders.text_encoder import TxtEmbeddings


class LFEncoder(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config
		self.img_embeddings = ImageEncoder(config)
		self.txt_embeddings = TxtEmbeddings(config)
		self.fusion_layer = FusionLayer(config)


	def forward(self, batch):
		# shape: [batch_size, 1, num_proposals, hidden_size]
		img_feats = self.img_embeddings(batch)

		txt_feats = self.txt_embeddings(batch, type='lf')

		# shape: [batch_size, num_rounds, hidden_size]
		fused_feats = self.fusion_layer(img_feats, txt_feats)

		return fused_feats


class FeatureAttention(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.attn_linear = nn.Linear(config['model']['hidden_size'], 1)
		self.dropout = nn.Dropout(p=config['model']['dropout'])

	def forward(self, ques_feats, attn_feats, attn_masks):
		# shape: [batch_size, num_rounds, num_feats, 1] dtype=uint8
		attn_masks = attn_masks[:, :, :, None].byte()

		# shape: [batch_size, num_rounds, num_feats, 1]
		attn_weights = self.attn_linear(self.dropout(attn_feats * ques_feats))

		attn_weights[~attn_masks] = -9999999.0

		# shape: [batch_size, num_rounds, num_feats, 1]
		attn_weights = F.softmax(attn_weights, dim=-2)

		# shape: [batch_size, num_rounds, hidden_size]
		attn_feats = (attn_weights * attn_feats).sum(dim=-2)

		# shape: [batch_size, num_rounds, 1, hidden_size]
		return attn_feats[:, :, None, :]


class FusionLayer(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config

		self.img_attn = FeatureAttention(config)

		self.is_bert = self.config['model']['encoder']['txt_embeddings']['type']

		if self.is_bert:
			self.hist_attn = FeatureAttention(config)

		self.fused_linear = nn.Linear(
				config['model']['hidden_size'] * 3,
				config['model']['hidden_size']
				)
		self.dropout = nn.Dropout(p=config['model']['dropout'])


	def forward(self, img_feats, txt_feats):
		# shape: [batch_size, num_rounds, 1, hidden_size]
		ques_feats = txt_feats['ques_feats']

		# shape: [batch_size, num_rounds, num_qas, hidden_size]
		hist_feats = txt_feats['hist_feats']

		# shape: [batch_size, num_rounds, num_proposals, hidden_size]
		img_feats = img_feats.repeat(1, ques_feats.size()[1], 1, 1)

		img_masks = torch.ones(img_feats.size()[:-1], device=img_feats.device)

		# shape: [batch_size, num_rounds, 1, hidden_size]
		img_feats = self.img_attn(ques_feats, img_feats, img_masks)

		if self.is_bert:
			hist_masks = txt_feats['hist_masks']
			# shape: [batch_size, num_rounds, 1, hidden_size]
			hist_feats = self.hist_attn(ques_feats, hist_feats, hist_masks)

		# shape: [batch_size, num_rounds, 1, hidden_size * 3]
		fused_feats = torch.cat((img_feats, ques_feats, hist_feats), dim=-1)

		# shape: [batch_size, num_rounds, 1, hidden_size]
		fused_feats = torch.tanh(self.fused_linear(self.dropout(fused_feats)))

		#shape: [batch_size, num_rounds, hidden_size]
		return fused_feats.squeeze(-2)



