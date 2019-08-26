import torch
import torch.nn as nn
from visdial.common.dynamic_rnn import DynamicRNN


class Decoder(nn.Module):

    def __init__(self, text_embeddings, disc_decoder=None, gen_decoder=None):
        super(Decoder, self).__init__()
        self.text_embeddings = text_embeddings
        self.disc_decoder = disc_decoder
        self.gen_decoder = gen_decoder

    def forward(self, batch, encoder_out):
        opt_scores = None
        ans_out_scores = None
        opts_out_scores = None
        bs, num_rounds, _ = batch['hist_tokens'].shape

        if self.disc_decoder is not None:
            opt_embeds = self.text_embeddings(batch['opts'])
            opt_scores = self.disc_decoder(opt_embeds, batch['opts_len'], encoder_out)
            opt_scores = opt_scores.view(bs, num_rounds, 100)

        if self.gen_decoder is not None:
            # shape [bs, num_hist, max_seq_len]
            if self.training:
                ans_in_embeds = self.text_embeddings(batch['ans_in'])
                ans_out_scores = self.gen_decoder(batch, ans_in_embeds, encoder_out)
                ans_out_scores = ans_out_scores.view(bs, num_rounds, -1)

            else:
                opts_in_embeds = self.text_embeddings(batch['opts_in'])
                opts_out_scores = self.gen_decoder(batch, opts_in_embeds, encoder_out)
                opts_out_scores = opts_out_scores.view(bs, num_rounds, 100)

        return {
            'opt_scores'     : opt_scores,
            'ans_out_scores' : ans_out_scores,
            'opts_out_scores': opts_out_scores
        }


class DiscriminativeDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.test_mode = self.config['model']['test_mode']

        self.word_embed = nn.Embedding(
            config['model']['vocab_size'],
            config['model']['embedding_size'],
            padding_idx=0,
        )

        self.option_rnn = nn.LSTM(
            config['model']['embedding_size'],
            config['model']['hidden_size'],
            num_layers=2,
            batch_first=True,
            dropout=config['model']['dropout'],
            bidirectional=config['model']['bidirectional']
        )

        self.option_linear = nn.Linear(config['model']['hidden_size'] * 2,
                                       config['model']['hidden_size'])

        # nn.init.kaiming_uniform_(self.option_linear.weight)
        # nn.init.constant_(self.option_linear.bias, 0)

        # Options are variable length padded sequences, use DynamicRNN.
        self.option_rnn = DynamicRNN(self.option_rnn)

    def forward(self, batch, encoder_output):
        """Given `encoder_output` + candidate option sequences, predict a score
        for each option sequence.
        Parameters
        ----------
        encoder_output: torch.Tensor
            Output from the encoder through its forward pass.
            (batch_size, num_rounds, lstm_hidden_size)
        """

        # shape: [BS, NR, NO, SEQ]
        options = batch["opts"]

        # batch_size, num_rounds, num_opts, seq_len
        BS, NR, NO, SEQ = options.size()
        HS = self.config['model']['hidden_size']

        # shape: [BS x NR x NO, SEQ]
        options = options.view(BS * NR * NO, SEQ)

        # shape: [BS, NR, NO]
        options_length = batch["opts_len"]

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

        # shape: [lstm_layers x bi, BS x NR x NO, HS]
        # shape: [lstm_layers x bi, BS x 1  x NO, HS] FOR TEST SPLIT,
        _, (nonzero_options_embed, _) = self.option_rnn(
            nonzero_options_embed, nonzero_options_length
        )

        # shape: [2, BS x NR x NO, HS]
        nonzero_options_embed = nonzero_options_embed[-2:]

        # shape: [BS x NR x NO, HS x 2]
        nonzero_options_embed = torch.cat([nonzero_options_embed[0], nonzero_options_embed[1]], dim=-1)

        # shape: [BS x NR x NO, HS]
        nonzero_options_embed = self.option_linear(nonzero_options_embed)

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

        if self.test_mode:
            scores = scores[:, batch['num_rounds'] - 1]

        return {'opt_scores': scores}


class MiscDecoder(nn.Module):

    def __init__(self, disc_decoder, gen_decoder):
        super(MiscDecoder, self).__init__()
        self.disc_decoder = disc_decoder
        self.gen_decoder = gen_decoder

    def forward(self, batch, encoder_output):
        output = {}
        if self.disc_decoder is not None:
            output['opt_scores'] = self.disc_decoder(batch, encoder_output)['opt_scores']

        if self.gen_decoder is not None:
            if self.training:
                output['ans_out_scores'] = self.gen_decoder(batch, encoder_output)['ans_out_scores']
            else:
                output['opts_out_scores'] = self.gen_decoder(batch, encoder_output)['opts_out_scores']

        return output


class GenerativeDecoder(nn.Module):
    def __init__(self, config):
        super(GenerativeDecoder, self).__init__()
        self.config = config
        self.test_mode = self.config['model']['test_mode']
        self.word_embed = nn.Embedding(
            config['model']['vocab_size'],
            config['model']['embedding_size'],
            padding_idx=0,
        )
        self.answer_rnn = nn.LSTM(
            config['model']['embedding_size'],
            config['model']['hidden_size'],
            num_layers=2,
            batch_first=True,
            dropout=config['model']['dropout']
        )

        self.lstm_to_words = nn.Linear(
            config['model']['hidden_size'], config['model']['vocab_size']
        )
        # nn.init.kaiming_uniform_(self.lstm_to_words.weight)
        # nn.init.constant_(self.lstm_to_words.bias, 0)

        self.dropout = nn.Dropout(p=config['model']['dropout'])
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, batch, encoder_output):
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
            # shape: [BS, NH, SEQ]
            ans_in = batch["ans_in"]
            (BS, NH, SEQ), HS = ans_in.size(), self.config['model']['hidden_size']

            # shape: [BS x NH, SEQ]
            ans_in = ans_in.view(BS * NH, SEQ)

            # shape: [BS x NH, SEQ, WE]
            ans_in_embed = self.word_embed(ans_in)

            # reshape encoder output to be set as initial hidden state of LSTM.
            # shape: [lstm_layers, BS x NH, HS]
            num_lstm_layers = 2
            init_hidden = encoder_output.view(1, BS * NH, -1).repeat(num_lstm_layers, 1, 1)

            init_cell = torch.zeros_like(init_hidden)

            # shape: [BS x NH, SEQ, HS]
            ans_out, (_, _) = self.answer_rnn(ans_in_embed, (init_hidden, init_cell))
            ans_out = self.dropout(ans_out)

            # shape: [BS, NH, SEQ, VC]
            return {'ans_out_scores' : self.lstm_to_words(ans_out).view(BS, NH, SEQ, -1)}

        else:
            opts_in = batch["opts_in"]
            target_opts_out = batch["opts_out"]

            if self.test_mode:
                # shape: [BS, NH, NO, SEQ]
                opts_in = opts_in[:, batch['num_rounds'] - 1]
                # shape: [BS x NH x NO, SEQ]
                target_opts_out = batch["opts_out"][:, batch['num_rounds'] - 1]

            BS, NH, NO, SEQ = opts_in.size()

            target_opts_out = target_opts_out.view(BS * NH * NO, -1)

            # shape: [BS x NH x NO, SEQ]
            opts_in = opts_in.view(BS * NH * NO, SEQ)

            # shape: [BS x NH x NO, WE]
            opts_in_embed = self.word_embed(opts_in)

            # reshape encoder output to be set as initial hidden state of LSTM.

            # shape: [BS, NH, 1, HS]
            init_hidden = encoder_output.view(BS, NH, 1, -1)

            # shape: [BS, NH, NO, HS]
            init_hidden = init_hidden.repeat(1, 1, NO, 1)

            # shape: [1, BS x NH x NO, HS]
            init_hidden = init_hidden.view(1, BS * NH * NO, -1)

            num_lstm_layers = 2
            # shape: [lstm_layers, BS x NH x NO, HS]
            init_hidden = init_hidden.repeat(num_lstm_layers, 1, 1)

            init_cell = torch.zeros_like(init_hidden)

            # shape: [BS x NH x NO, SEQ, HS]
            opts_out, (_, _) = self.answer_rnn(opts_in_embed, (init_hidden, init_cell))

            # shape: [BS x NH x NO, SEQ, VC]
            opts_word_scores = self.logsoftmax(self.lstm_to_words(opts_out))

            # shape: [BS x NH x NO, SEQ]
            opts_out_scores = torch.gather(opts_word_scores, -1, target_opts_out.unsqueeze(-1)).squeeze()
            # ^ select the scores for target word in [vocab vector] of each word

            # shape: [BS x NH x NO, SEQ] <- remove the <PAD> word
            opts_out_scores = (opts_out_scores * (target_opts_out > 0).float())

            # sum all the scores for each word in the predicted answer -> final score
            # shape: [BS x NH x NO]
            opts_out_scores = torch.sum(opts_out_scores, dim=-1)

            # shape: [BS, NH, NO]
            opts_out_scores = opts_out_scores.view(BS, NH, NO)

            return {'opts_out_scores' : opts_out_scores}


            # else:
            #     # shape: [BS, NH, NO, SEQ]
            #     opts_in = batch["opts_in"]
            #     last_round = batch['num_rounds'] - 1
            #
            #     # shape: [BS, NO, SEQ]
            #     opts_in = opts_in[:, last_round]
            #     print('opts_in shape', opts_in.shape)
            #     BS, _, NO, SEQ = opts_in.size()
            #
            #     # shape: [BS x NO, SEQ]
            #     opts_in = opts_in.view(BS * NO, SEQ)
            #
            #     # shape: [BS x NO, WE]
            #     opts_in_embed = self.word_embed(opts_in)
            #
            #     # reshape encoder output to be set as initial hidden state of LSTM.
            #
            #     # shape: [BS, 1, HS]
            #     init_hidden = encoder_output.view(BS, 1, -1)
            #
            #     # shape: [BS, NO, HS]
            #     init_hidden = init_hidden.repeat(1, NO, 1)
            #
            #     num_lstm_layers = 2
            #     # shape: [lstm_layers, BS x NO, HS]
            #     init_hidden = init_hidden.repeat(num_lstm_layers, 1, 1)
            #     init_cell = torch.zeros_like(init_hidden)
            #
            #     # shape: [BS x NO, SEQ, HS]
            #     opts_out, (_, _) = self.answer_rnn(opts_in_embed, (init_hidden, init_cell))
            #
            #     # shape: [BS x NO, SEQ, VC]
            #     opts_word_scores = self.logsoftmax(self.lstm_to_words(opts_out))
            #
            #     # shape: [BS x NO, SEQ]
            #     target_opts_out = batch["opts_out"][:, last_round].view(BS * NO, -1)
            #
            #     # shape: [BS x NO, SEQ]
            #     opts_out_scores = torch.gather(opts_word_scores, -1, target_opts_out.unsqueeze(-1)).squeeze()
            #     # ^ select the scores for target word in [vocab vector] of each word
            #
            #     # shape: [BS x NO, SEQ] <- remove the <PAD> word
            #     opts_out_scores = (opts_out_scores * (target_opts_out > 0).float())
            #
            #     # sum all the scores for each word in the predicted answer -> final score
            #     # shape: [BS x NO]
            #     opts_out_scores = torch.sum(opts_out_scores, dim=-1)
            #
            #     # shape: [BS, NO]
            #     opts_out_scores = opts_out_scores.view(BS, NO)
            #     return {'opts_out_scores' : opts_out_scores}
