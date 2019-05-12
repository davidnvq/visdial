import torch
from torch import nn


class GenerativeDecoder(nn.Module):
	def __init__(self, config, vocabulary):
		super().__init__()
		self.config = config

		self.word_embed = nn.Embedding(
				len(vocabulary),
				config["word_embedding_size"],
				padding_idx=vocabulary.PAD_INDEX,
				)
		self.answer_rnn = nn.LSTM(
				config["word_embedding_size"],
				config["lstm_hidden_size"],
				config["lstm_num_layers"],
				batch_first=True,
				dropout=config["dropout"],
				)

		self.lstm_to_words = nn.Linear(
				self.config["lstm_hidden_size"], len(vocabulary)
				)

		self.dropout = nn.Dropout(p=config["dropout"])
		self.logsoftmax = nn.LogSoftmax(dim=-1)

	def forward(self, encoder_output, batch):
		"""Given `encoder_output`, learn to autoregressively predict
		ground-truth answer word-by-word during training.

		During evaluation, assign log-likelihood scores to all answer options.

		Parameters
		----------
		encoder_output: torch.Tensor
			Output from the encoder through its forward pass.
			(batch_size, num_rounds, lstm_hidden_size)
		"""
		# make it single contiguous chunk of memory
		self.answer_rnn.flatten_parameters()

		if self.training:
			# shape: [BS, NR, SEQ]
			ans_in = batch["ans_in"]
			(BS, NR, SEQ), HS = ans_in.size(), self.config['lstm_hidden_size']

			# shape: [BS x NR, SEQ]
			ans_in = ans_in.view(BS * NR, SEQ)

			# shape: [BS x NR, SEQ, WE]
			ans_in_embed = self.word_embed(ans_in)

			# reshape encoder output to be set as initial hidden state of LSTM.
			# shape: [lstm_layers, BS x NR, HS]
			init_hidden = encoder_output.view(1, BS * NR, -1).repeat(self.config['lstm_num_layers'], 1, 1)

			init_cell = torch.zeros_like(init_hidden)

			# shape: [BS x NR, SEQ, HS]
			ans_out, (_, _) = self.answer_rnn(ans_in_embed, (init_hidden, init_cell))
			ans_out = self.dropout(ans_out)

			# shape: [BS x NR, SEQ, VC]
			ans_word_scores = self.lstm_to_words(ans_out)
			return ans_word_scores

		else:
			# shape: [BS, NR, NO, SEQ]
			ans_in = batch["opt_in"]
			BS, NR, NO, SEQ = ans_in.size()

			# shape: [BS x NR x NO, SEQ]
			ans_in = ans_in.view(BS * NR * NO, SEQ)

			# shape: [BS x NR x NO, WE]
			ans_in_embed = self.word_embed(ans_in)

			# reshape encoder output to be set as initial hidden state of LSTM.

			# shape: [BS, NR, 1, HS]
			init_hidden = encoder_output.view(BS, NR, 1, -1)

			# shape: [BS, NR, NO, HS]
			init_hidden = init_hidden.repeat(1, 1, NO, 1)

			# shape: [1, BS x NR x NO, HS]
			init_hidden = init_hidden.view(1, BS * NR * NO, -1)

			# shape: [lstm_layers, BS x NR x NO, HS]
			init_hidden = init_hidden.repeat(self.config["lstm_num_layers"], 1, 1)

			init_cell = torch.zeros_like(init_hidden)

			# shape: [BS x NR x NO, SEQ, HS]
			ans_out, (_, _) = self.answer_rnn(ans_in_embed, (init_hidden, init_cell))

			# shape: [BS x NR x NO, SEQ, VC]
			ans_word_scores = self.logsoftmax(self.lstm_to_words(ans_out))

			# shape: [BS x NR x NO, SEQ]
			target_ans_out = batch["opt_out"].view(BS * NR * NO, -1)

			# shape: [BS x NR x NO, SEQ]
			ans_word_scores = torch.gather(ans_word_scores, -1, target_ans_out.unsqueeze(-1)).squeeze()
			# ^ select the scores for target word in [vocab vector] of each word

			# shape: [BS x NR x NO, SEQ] <- remove the <PAD> word
			ans_word_scores = (ans_word_scores * (target_ans_out > 0).float().cuda())

			# sum all the scores for each word in the predicted answer -> final score
			# shape: [BS x NR x NO]
			ans_scores = torch.sum(ans_word_scores, dim=-1)

			# shape: [BS, NR, NO]
			ans_scores = ans_scores.view(BS, NR, NO)

			return ans_scores
