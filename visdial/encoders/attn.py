import torch
from torch import nn
from torch.nn import functional as F
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from visdial.utils import DynamicRNN


def clones(module, N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask_v=None, dropout=None):
	"Compute 'Scaled Dot Product Attention'"
	d_k = query.size(-1)
	num_heads = query.size(1)
	batch_size = query.size(0)

	# key.transpose(-2, -1)
	# shape: [BS x NR, NHeads, Nseq2, DK] -> [BS x NR, NHeads, DK, Nseq2]

	# shape: [BS x NR, NHeads, Nseq1, DK]  x [BS x NR, NHeads, DK, Nseq1]
	# shape: [BS x NR, NHeads, Nseq1, Nseq2] <- scores
	scores = torch.matmul(query, key.transpose(-2, -1))
	scores /= math.sqrt(d_k)

	if mask_v is not None:
		# shape: # [BS, Nseq2] -> [BS, 1, 1, Nseq2]
		mask_v = mask_v[:, None, None, :]
		# shape: [BS x NR, NHeads, Nseq1, Nseq2]
		scores = scores.masked_fill(mask_v == 0, -1e9)
	# shape: [BS x NR, NHeads, Nseq1, Nseq2]
	p_attn = F.softmax(scores, dim=-1)

	if dropout is not None:
		p_attn = dropout(p_attn)

	# shape: [BS x NR, NHeads, Nseq1, Nseq2] x [BS x NR, NHeads, Nseq2, DK]
	# shape: [BS x NR, NHeads, Nseq1, DK]
	attn = torch.matmul(p_attn, value)

	# shape: [BS x NR, Nseq1, Nheads, DK]
	attn = attn.transpose(1, 2).contiguous()
	# shape: [BS x NR, Nseq1, HS]
	attn = attn.view(batch_size, -1, num_heads * d_k)
	return attn, p_attn


class LayerNorm(nn.Module):
	"Construct a layernorm module (See citation for details)."

	def __init__(self, d_model, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(d_model))
		self.b_2 = nn.Parameter(torch.zeros(d_model))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MultiHeadedAttention(nn.Module):
	def __init__(self, d_model, num_heads, dropout=0.1):
		super(MultiHeadedAttention, self).__init__()
		self.linears_i = clones(nn.Linear(d_model, d_model), 3)
		self.linears_q = clones(nn.Linear(d_model, d_model), 3)
		self.linears_p = clones(nn.Linear(d_model, d_model), 2)  # projections for img, hist, quest

		self.num_heads = num_heads
		self.d_model = d_model
		self.d_k = d_model // num_heads
		self.dropout = nn.Dropout(p=dropout)

	def linear_and_transpose(self, linear, x):
		# shape: [BS x NR]
		batch_size = x.size(0)
		# shape: [BS x NR, NP, HS]
		out = linear(x)
		# shape: [BS x NR, NP, NHeads, DK]
		out = out.view(batch_size, -1, self.num_heads, self.d_k)
		# shape: [BS x NR, NHeads, NP, DK]
		out = out.transpose(1, 2)
		return out

	def forward(self, img_que, masks):
		# shape: [BS x NR, NHeads, NP, DK]
		iq, ik, iv = [self.linear_and_transpose(l, img) for l, img in zip(self.linears_i, [img_que[0]] * 3)]
		# shape: [BS x NR, NHeads, SEQ, DK]
		qq, qk, qv = [self.linear_and_transpose(l, que) for l, que in zip(self.linears_q, [img_que[1]] * 3)]

		# shape: [BS x NR, NSeqImg, HS]
		q2i_attn, _ = attention(iq, qk, qv, mask_v=masks[1], dropout=self.dropout)

		# shape: [BS x NR, NSeqQue, HS]
		i2q_attn, _ = attention(qq, ik, iv, mask_v=masks[0], dropout=self.dropout)

		# shape: [BS x NR, NSeqImg, HS], [BS x NR, NSeqQue, HS]
		return [l(x) for l, x in zip(self.linears_p, [q2i_attn, i2q_attn])]


class SublayerConnection(nn.Module):
	"""
	A residual connection followed by a layer norm.
	Note for code simplicity the norm is first as opposed to last.
	"""

	def __init__(self, d_model, dropout):
		super(SublayerConnection, self).__init__()
		self.norms = clones(LayerNorm(d_model), 2)
		self.dropout = nn.Dropout(dropout)

	def forward(self, img_que, sublayer, masks=None):
		"Apply residual connection to any sublayer with the same size."
		# shape: [BS x NR, Nseq, HS]
		normed_i = self.norms[0](img_que[0])
		normed_q = self.norms[1](img_que[1])

		# shape: [BS x NR, Nseq, HS]
		out_i, out_q = sublayer((normed_i, normed_q), masks=masks)
		# shape: [BS x NR, NseqImg, HS], [BS x NR, NseqQue, HS]
		return [img_que[0] + self.dropout(out_i),
		        img_que[1] + self.dropout(out_q)]


class PositionwiseFeedForward(nn.Module):
	"""	Implements FFN equation.
	FFN(x) = relu(0, xW1 + b1)W2 + b2
	"""

	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.linears_1 = clones(nn.Linear(d_model, d_ff), 2)
		self.linears_2 = clones(nn.Linear(d_ff, d_model), 2)
		self.dropout = nn.Dropout(dropout)

	def forward(self, img_que, masks=None):
		# shape: [BS x NR, NseqImg, HS], [BS x NR, NseqQue, HS]
		return [self.linears_2[0](self.dropout(F.relu(self.linears_1[0](img_que[0])))),
		        self.linears_2[1](self.dropout(F.relu(self.linears_1[1](img_que[1]))))]


class EncoderLayer(nn.Module):
	"Encoder is made up of self-attn and feed forward (defined below)"

	def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
		super(EncoderLayer, self).__init__()
		self.attn_layer = MultiHeadedAttention(d_model, num_heads, dropout)
		self.ffwd_layer = PositionwiseFeedForward(d_model, d_ff, dropout)
		self.sublayers = clones(SublayerConnection(d_model, dropout), 2)
		self.dropout = nn.Dropout(dropout)

	def forward(self, img_que, masks):
		"Follow Figure 1 (left) for connections."
		img_que = self.sublayers[0](img_que, self.attn_layer, masks=masks)
		img_que = self.sublayers[1](img_que, self.ffwd_layer, masks=None)
		return [self.dropout(x) for x in img_que]


class SelfAttention(nn.Module):

	def __init__(self, d_model):
		super(SelfAttention, self).__init__()
		self.attn = nn.Sequential(
				nn.Linear(d_model, 512),
				nn.ReLU(inplace=True),
				nn.Linear(512, 1))

	def forward(self, x, mask_x):
		# shape: [BS x NR, Nseq,   ] <- mask_x
		# shape: [BS x NR, Nseq, HS] <- x

		# shape: [BS x NR, Nseq]
		logits = self.attn(x).squeeze(-1)

		if mask_x is not None:
			# shape: [BS x NR, Nseq]
			logits = logits.masked_fill(mask_x == 0, -1e9)

		# shape: [BS x NR, Nseq]
		return F.softmax(logits, dim=-1)


class AttentionEncoder(nn.Module):
	def __init__(self, config, vocabulary):
		super().__init__()
		self.config = config
		WE = config["word_embedding_size"]
		HS = config["lstm_hidden_size"]
		num_lstm_layers = config["lstm_num_layers"]
		dropout = config["dropout"]
		num_attns = config["num_attns"]

		self.word_embed = nn.Embedding(len(vocabulary), WE, 0)
		self.his_rnn = nn.LSTM(WE, HS, num_lstm_layers,
		                       batch_first=True,
		                       dropout=dropout,
		                       bidirectional=True)
		self.his_rnn = DynamicRNN(self.his_rnn)
		self.que_rnn = nn.LSTM(WE, HS, num_lstm_layers,
		                       batch_first=True,
		                       dropout=dropout,
		                       bidirectional=True)
		self.que_rnn = DynamicRNN(self.que_rnn)

		self.his_linear = nn.Linear(HS * 2, HS)
		self.que_linear = nn.Linear(HS * 2, HS)
		self.img_embed_layer = nn.Linear(config["img_feature_size"], HS)

		self.encoders = clones(EncoderLayer(HS, HS, 8, dropout=dropout), num_attns)
		self.selfattn = clones(SelfAttention(HS), 2)
		self.fused_embed_layer = nn.Linear(HS * 3, HS)

		self.norms = clones(LayerNorm(HS), 2)
		self.dropout = nn.Dropout(p=config["dropout"])


	def forward(self, batch, debug=False):
		# shape: (batch_size, num_proposals, img_feature_size) - RCNN features
		# shape: [BS, NP, IS]
		img = batch["img_feat"]

		# shape: [BS, 10, SEQ]
		ques = batch["ques"]

		# shape: [BS, 10, SEQ x 2 x 10] <- concatenated q & a * 10 rounds
		hist = batch["hist"]

		# num_rounds = 10, even for test (padded dialog rounds at the end)
		(BS, NR, SEQ), HS = ques.size(), self.config["lstm_hidden_size"]
		NP, IS = img.size(1), img.size(2)

		# mask
		# shape: [BS x NR, NP]
		img_mask = torch.ones(BS * NR, NP).cuda()

		# shape: [BS x NR, SEQ]
		# que_mask = torch.ones(BS * NR, SEQ).cuda()
		que_mask = (ques.view(BS * NR, SEQ) > 0.0).cuda()

		# embed questions
		# shape: [BS x NR, SEQ]
		ques = ques.view(BS * NR, SEQ)

		# shape: [BS x NR, SEQ, WE]
		que_embed = self.word_embed(ques)

		# shape: [BS x NR, SEQ, HS x 2]
		que_len = batch['ques_len'].view(-1, )
		que_embed, (_, _) = self.que_rnn(que_embed, que_len)

		# embed history
		# shape: [BS x NR, SEQ x 20]
		hist = hist.view(BS * NR, SEQ * 20)

		# shape: [BS x NR, SEQ x 20, WE]
		his_embed = self.word_embed(hist)

		# shape: [num_layers x 1, BS x NR, HS] in 1 directional
		# shape: [num_layers x 2, BS x NR, HS] in 2 directionals
		his_len = batch['hist_len'].view(-1)

		_, (his_embed, _) = self.his_rnn(his_embed, his_len)

		# shape: [BS x NR, SEQ, HS]
		que_embed = self.que_linear(que_embed)

		# shape: [2, BS x NR, HS]
		his_embed = his_embed[-2:]
		# shape: [BS x NR, HS x 2]
		his_embed = torch.cat([his_embed[0], his_embed[1]], dim=-1)
		# shape: [BS x NR, HS]
		his_embed = self.his_linear(his_embed)

		# project down image features and ready for attention
		# shape: [BS, NP, HS]
		img_embed = F.relu(self.img_embed_layer(img))

		# shape: [BS, NP, HS] -> [BS, NR, NP, HS]
		img_embed = img_embed[:, None, :, :].repeat(1, NR, 1, 1)

		# shape: [BS x NR, NP, HS]
		img_embed = img_embed.view(BS * NR, NP, HS)

		# NOW pass to encoder layer
		img_que = [img_embed, que_embed]
		masks = [img_mask, que_mask]

		for encoder in self.encoders:
			img_que = encoder(img_que, masks)

		# shape: [BS x NR, NP, HS]
		img_embed = self.norms[0](img_que[0])
		# shape: [BS x NR, NP] -> [BS x NR, 1, NP]
		img_probs = self.selfattn[0](img_embed, img_mask)[:, None, :]

		# shape: [BS x NR, 1, HS] -> [BS x NR, HS]
		img_embed = torch.matmul(img_probs, img_embed).squeeze(-2)

		# shape: [BS x NR, SEQ, HS]
		que_embed = self.norms[1](img_que[1])

		# shape: [BS x NR, SEQ] -> [BS x NR, SEQ, 1]
		que_probs = self.selfattn[1](que_embed, que_mask)[:, None, :]
		# shape: [BS x NR, HS]
		que_embed = torch.matmul(que_probs, que_embed).squeeze(-2)

		# shape: [BS x NR, HS x 3]
		fused_vector = torch.cat((img_embed, que_embed, his_embed), -1)
		fused_vector = self.dropout(fused_vector)

		# shape: [BS x NR, HS]
		fused_embed = torch.tanh(self.fused_embed_layer(fused_vector))

		# shape: [BS, NR, HS]
		fused_embed = fused_embed.view(BS, NR, HS)

		if debug:
			return fused_embed, None
		return fused_embed
