import torch
from torch import nn

class GenDecoder(nn.Module):

    def __init__(self, hidden_size, vocab_size, num_lstm_layers, dropout, **kwargs):

        super(GenDecoder, self).__init__()
        self.ans_rnn = nn.LSTM(hidden_size,
                               hidden_size,
                               num_layers=num_lstm_layers,
                               batch_first=True,
                               dropout=dropout)
        self.lstm_to_words = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.logsoftmax = nn.LogSoftmax(dim=-1)


    def forward(self, batch, ans_in_embed, encoder_out):
        """
        :param answer_embed: shape [bs, num_hist, max_seq_len, hidden_size]
        :param encoder_out:  shape [bs * num_hist, hidden_size]
        :return:
        """
        self.ans_rnn.flatten_parameters()
        num_lstm_layers = self.ans_rnn.num_layers

        if self.training: # training
            print("ans_in_embed", ans_in_embed.size())

            bs, num_hist, max_seq_len, hidden_size = ans_in_embed.size()

            ans_in_embed = ans_in_embed.view(bs * num_hist, max_seq_len, hidden_size)

            # shape [lstm_layers, bs * num_hist, hidden_size]
            init_hidden = encoder_out.view(1,
                                           bs * num_hist,
                                           hidden_size).repeat(num_lstm_layers, 1, 1)
            print("init_hidden0", init_hidden.size())

            init_cell = torch.zeros_like(init_hidden)

            # shape [bs * num_hist, max_seq_len, hidden_size]
            ans_out, (_, _) = self.ans_rnn(ans_in_embed, (init_hidden, init_cell))

            ans_out = self.dropout(ans_out)
            ans_word_scores = self.lstm_to_words(ans_out)
            return ans_word_scores

        else: # validation and testing
            bs, num_hist, num_opts, max_seq_len, hidden_size = ans_in_embed.size()
            # shape [bs, num_hist, num_opts, max_seq_len, hidden_size]
            opts_in = ans_in_embed

            # shape [bs * num_hist * num_opts, max_seq_len, hidden_size]
            opts_in = opts_in.view(bs * num_hist * num_opts, max_seq_len, hidden_size)

            # shape [bs, num_hist, 1, hidden_size]
            init_hidden = encoder_out.view(bs, num_hist, 1, -1)
            print("init_hidden1", init_hidden.size())

            # shape [bs, num_hist, num_opts, hidden_size]
            init_hidden = init_hidden.repeat(1, 1, num_opts, 1)
            print("init_hidden2", init_hidden.size())

            # shape [lstm_layers, bs * num_hist * num_opts, hidden_size]
            init_hidden = init_hidden.view(1,
                                           bs * num_hist * num_opts,
                                           hidden_size)
            print("init_hidden3", init_hidden.size())

            init_hidden = init_hidden.repeat(num_lstm_layers, 1, 1)
            print("init_hidden4", init_hidden.size())

            init_cell = torch.zeros_like(init_hidden)

            # shape [bs * num_hist * num_opts, max_seq_len, hidden_size]
            opts_out, (_, _) = self.ans_rnn(opts_in, (init_hidden, init_cell))

            # shape [bs * num_hist * num_opts, max_seq_len, vocab_size]
            opts_out_scores = self.logsoftmax(self.lstm_to_words(opts_out))

            # shape [bs * num_hist * num_opts, max_seq_len]
            target_opts_out = batch['opts_out'].view(bs * num_hist * num_opts, -1)

            # shape [bs * num_hist * num_opts, max_seq_len]
            target_opts_out = torch.gather(opts_out_scores, -1, target_opts_out.unsqueeze(-1)).squeeze()
            # ^ select the scores for target word in [vocab vector] of each word

            # shape [bs * num_hist * num_opts, max_seq_len] <- remove <PAD> token
            opts_out_scores = (opts_out_scores * (target_opts_out > 0).float())#.cuda()

            # sum all the scores for each word in the predicted answer -> final score
            # shape [bs * num_hist * num_opts]
            opts_out_scores = opts_out_scores.view(bs * num_hist, num_opts)
            return opts_out_scores
