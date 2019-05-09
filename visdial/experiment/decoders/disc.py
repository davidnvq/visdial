import torch
from torch import nn

from visdial.utils import DynamicRNN


class DiscriminativeDecoder(nn.Module):
	def __init__(self, config, vocabulary):
		super().__init__()
		self.config = config

		self.word_embed = nn.Embedding(
				len(vocabulary),
				config["word_embedding_size"],
				padding_idx=vocabulary.PAD_INDEX,
				)
		self.option_rnn = nn.LSTM(
				config["word_embedding_size"],
				config["lstm_hidden_size"],
				config["lstm_num_layers"],
				batch_first=True,
				dropout=config["dropout"],
				)

		# Options are variable length padded sequences, use DynamicRNN.
		self.option_rnn = DynamicRNN(self.option_rnn)

	def forward(self, encoder_output, batch):
		"""Given `encoder_output` + candidate option sequences, predict a score
		for each option sequence.

		Parameters
		----------
		encoder_output: torch.Tensor
			Output from the encoder through its forward pass.
			(batch_size, num_rounds, lstm_hidden_size)
		"""
		# shape: [BS, NR, NO, SEQ]
		options = batch["opt"]
		# batch_size, num_rounds, num_opts, seq_len
		BS, NR, NO, SEQ = options.size()
		HS = self.config["lstm_hidden_size"]

		# shape: [BS x NR x NO, SEQ]
		options = options.view(BS * NR * NO, SEQ)

		# shape: [BS, NR, NO]
		options_length = batch["opt_len"]

		# shape: [BS x NR x NO]
		options_length = options_length.view(BS * NR * NO)

		# Pick options with non-zero length (relevant for test split).
		# shape: [BS x (nR x NO)] <- nR ~= 1 or 10 for test: nR = 1, for train, val nR = 10
		nonzero_options_length_indices = options_length.nonzero().squeeze()

		# shape: [BS x (nR x NO)]
		nonzero_options_length = options_length[nonzero_options_length_indices]

		# shape: [BS x (nR x NO)]
		nonzero_options = options[nonzero_options_length_indices]

		# shape: [BS x NR x NO, SEQ, WE]
		# shape: [BS x 1  x NO, SEQ, WE] <- FOR TEST SPLIT
		nonzero_options_embed = self.word_embed(nonzero_options)

		# shape: [BS x NR x NO, HS]
		# shape: [BS x 1  x NO, HS] FOR TEST SPLIT,
		_, (nonzero_options_embed, _) = self.option_rnn(
				nonzero_options_embed, nonzero_options_length
				)

		# shape: [BS x NR x NO, HS] <- move back to standard for TEST split
		options_embed = torch.zeros(BS * NR * NO, HS, device=options.device)

		# shape: [BS x NR x NO, HS]
		options_embed[nonzero_options_length_indices] = nonzero_options_embed

		# TODONE: these lines are the same
		# shape: [BS, NR, SEQ] -> [BS, NR, SEQ, 1]
		encoder_output = encoder_output.unsqueeze(-1)

		# shape: [BS, NR, NO, SEQ]
		options_embed = options_embed.view(BS, NR, NO, -1)

		# shape: [BS, NR, NO, 1]
		scores = torch.matmul(options_embed, encoder_output)

		# shape: [BS, NR, NO]
		scores = scores.squeeze(-1)

		return scores
