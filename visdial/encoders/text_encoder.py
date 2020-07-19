import torch
import torch.nn as nn
from visdial.common import PositionalEmbedding, DynamicRNN, check_flag


class TextEncoder(nn.Module):
    def __init__(self, config, hist_encoder, ques_encoder):
        super(TextEncoder, self).__init__()
        self.text_embedding = nn.Embedding(config['model']['txt_vocab_size'],
                                           config['model']['txt_embedding_size'],
                                           padding_idx=0)

        self.hist_encoder = hist_encoder
        self.ques_encoder = ques_encoder

    def forward(self, batch, test_mode=False):
        """
        Arguments
        ---------
        batch: Dictionary
            This provides a dictionary of inputs.
        test_mode: Boolean
            Whether the forward is performed on test data
        Returns
        -------
        output :  A tuples of the following:
            hist: torch.FloatTensor
                The representation of history utility
                Shape [batch_size * NH, T, hidden_size]
            ques: torch.FloatTensor
                The representation of question utility
                Shape [batch_size * NH, N, hidden_size]
            hist_mask: torch.LongTensor
                The mask of history utility
                Shape [batch_size * NH, T]
            ques_mask: torch.LongTensor
                The mask of question utility
                Shape [batch_size * NH, N]
        """

        # ques_tokens: shape [BS, num_hist, N]
        # hist_tokens: shape [BS, num_hist, num_rounds, N]
        ques_tokens = batch['ques_tokens']
        hist_tokens = batch['hist_tokens']

        # ques_len: shape [BS, num_hist]
        # hist_len: shape [BS, num_hist, num_rounds]
        ques_len = batch['ques_len']
        hist_len = batch['hist_len']

        # ques: shape [BS, num_hist, N, embedding_size(or hidden_size)]
        # hist: shape [BS, num_hist, num_rounds, N, embedding_size(or hidden_size)]
        ques = self.text_embedding(ques_tokens)
        hist = self.text_embedding(hist_tokens)

        # ques: shape [BS x NH, N, hidden_size]
        # hist: shape [BS x NH, num_rounds, hidden_size]
        # ques_mask: shape [BS x NH, N]
        # hist_mask: shape [BS x NH, num_rounds]
        ques, ques_mask = self.ques_encoder(ques, ques_len, test_mode=test_mode)
        hist, hist_mask = self.hist_encoder(hist, hist_len, test_mode=test_mode)
        return hist, ques, hist_mask, ques_mask


class QuesEncoder(nn.Module):

    def __init__(self, config):
        super(QuesEncoder, self).__init__()

        self.config = config

        self.ques_linear = nn.Linear(config['model']['hidden_size'] * 2,
                                     config['model']['hidden_size'])

        self.ques_lstm = DynamicRNN(nn.LSTM(config['model']['txt_embedding_size'],
                                            config['model']['hidden_size'],
                                            num_layers=2,
                                            bidirectional=config['model']['txt_bidirectional'],
                                            batch_first=True))

        if config['model']['txt_has_layer_norm']:
            self.layer_norm = nn.LayerNorm(config['model']['hidden_size'])

        if config['model']['txt_has_pos_embedding']:
            self.pos_embedding = PositionalEmbedding(config['model']['hidden_size'],
                                                     config['dataset']['max_seq_len'])

        self.config = config

    def forward(self, ques, ques_len, test_mode=False):
        """
        Arguments
        ---------
        ques: torch.FloatTensor
            The embedding of question tokens
            Shape [batch_size, num_hist, N, embedding_size]
        test_mode: Boolean
            Whether the forward is performed on test data
        Returns
        -------
        output :  A tuples of the following:
            ques: torch.FloatTensor
                The representation of question utility
                Shape [batch_size * num_hist, N, hidden_size]
            ques_mask: torch.LongTensor
                The mask of question utility
                Shape [batch_size x NH, N]
        """
        # for test only
        if self.config['model']['test_mode'] or test_mode:
            # get only the last question
            last_idx = (ques_len > 0).sum()
            ques = ques[:, last_idx - 1:last_idx]
            ques_len = ques_len[:, last_idx - 1:last_idx]

        bs, num_hist, max_seq_len, embedding_size = ques.size()

        # shape [BS x NH, N, hidden_size]
        ques = ques.view(bs * num_hist, max_seq_len, embedding_size)

        # shape [BS * num_hist]
        ques_len = ques_len.view(bs * num_hist)

        # shape [BS x NH, N]
        ques_mask = torch.arange(max_seq_len, device=ques.device).repeat(bs * num_hist, 1)
        ques_mask = ques_mask < ques_len.unsqueeze(-1)

        if isinstance(self.ques_lstm, DynamicRNN):
            # LSTM
            if not self.ques_lstm.bidirectional:
                # shape: ques [BS x NH, N, hidden_size]
                ques, (_, _) = self.ques_lstm(ques, ques_len)

                # shape: ques [BS x NH, N, hidden_size]
                # shape: ques [BS x NH, N,]
                return ques, ques_mask.long()

            # BiLSTM
            else:
                # [BS x NH, SEQ, HS x 2]
                ques, (_, _) = self.ques_lstm(ques, ques_len)

                # [BS x NH, SEQ, HS]
                ques = self.ques_linear(ques)
                if self.config['model']['txt_has_pos_embedding']:
                    ques = ques + self.pos_embedding(ques)

                if self.config['model']['txt_has_layer_norm']:
                    ques = self.layer_norm(ques)

                return ques, ques_mask.long()


class HistEncoder(nn.Module):

    def __init__(self, config):
        super(HistEncoder, self).__init__()
        self.config = config

        self.hist_linear = nn.Linear(config['model']['hidden_size'] * 2,
                                     config['model']['hidden_size'])

        self.hist_lstm = DynamicRNN(nn.LSTM(config['model']['txt_embedding_size'],
                                            config['model']['hidden_size'],
                                            num_layers=2,
                                            bidirectional=config['model']['txt_bidirectional'],
                                            batch_first=True))

        if config['model']['txt_has_layer_norm']:
            self.layer_norm = nn.LayerNorm(config['model']['hidden_size'])

        if config['model']['txt_has_pos_embedding']:
            self.pos_embedding = PositionalEmbedding(config['model']['hidden_size'],
                                                     max_len=10)

        self.config = config
        self.hidden_size = self.config['model']['hidden_size']

    def forward(self, hist, hist_len, test_mode=False):
        """
        Arguments
        ---------
        hist: torch.FloatTensor
            The embedding of question tokens
            Shape [batch_size, num_hist, T, embedding_size]
        hist_len: torch.LongTensor
            The length of each dialog history round
        test_mode: Boolean
            Whether the forward is performed on test data
        Returns
        -------
        output :  A tuples of the following:
            hist: torch.FloatTensor
                The representation of history utility
                Shape [batch_size x NH, T, hidden_size]
            hist_mask: torch.LongTensor
                The mask of history utility
                Shape [batch_size x NH, T]
        """

        bs, num_rounds, max_seq_len, embedding_size = hist.size()

        if self.config['dataset']['concat_hist']:
            # for test only

            bs, num_hist, max_seq_len, embedding_size = hist.size()

            # shape [BS x NH, 2N, hidden_size]
            hist = hist.view(bs * num_hist, max_seq_len, embedding_size)

            # shape [BS * num_hist]
            hist_len = hist_len.view(bs * num_hist)

            # shape [BS x NH, 2N]
            hist_mask = torch.arange(max_seq_len, device=hist.device).repeat(bs * num_hist, 1)
            hist_mask = hist_mask < hist_len.unsqueeze(-1)

            if isinstance(self.hist_lstm, DynamicRNN):
                # [BS x NH, SEQ, HS x 2]
                hist, (_, _) = self.hist_lstm(hist, hist_len)

                # [BS x NH, SEQ, HS]
                hist = self.hist_mlp(hist)
                if self.config['model']['txt_has_pos_embedding']:
                    hist = hist + self.pos_embedding(hist)

                if self.config['model']['txt_has_layer_norm']:
                    hist = self.layer_norm(hist)
                return (hist,  # shape: [BS x NH, T, hidden_size]
                        hist_mask.long(),  # shape [BS x NH, T]
                        )
        else:
            # shape [BS * num_rounds, 2N, hidden_size]
            hist = hist.view(bs * num_rounds, max_seq_len, embedding_size)

            # shape [BS * num_rounds]
            hist_len = hist_len.view(bs * num_rounds)

            if self.config['model']['test_mode'] or test_mode:
                num_hist = 1
                round_mask = torch.ones(bs, num_hist, num_rounds, 1, device=hist.device)
                hist_mask = torch.ones(bs * num_hist, num_rounds, device=hist.device)
            else:
                num_hist = 10
                # shape [10, 10]
                MASK = torch.tensor([
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ], device=hist.device)

                # shape [BS, NH, T, 1]
                round_mask = MASK[None, :, :, None].repeat(bs, 1, 1, 1)

                hist_mask = MASK[None, :, :].repeat(bs, 1, 1).view(bs * num_hist, num_rounds)

            if isinstance(self.hist_lstm, DynamicRNN):

                if not self.hist_lstm.bidirectional:  # LSTM
                    # shape: [num_layers, BS, HS] if Not bidirectional
                    # shape: hn = [num_layers, bs * num_rounds, hidden_size]
                    y, (hn, cn) = self.hist_lstm(hist, hist_len)

                    # shape: [BS * num_rounds, hidden_size]
                    hist = hn[-1]

                    # shape: [BS, num_rounds, hidden_size]
                    hist = hist.view(bs, num_rounds, self.hidden_size)

                else:  # BiLSTM

                    # hn [num_layers x 2 (bidirectional), BS x NR, HS]
                    y, (hn, cn) = self.hist_lstm(hist, hist_len)

                    # shape: [2, BS x NR, HS]
                    hn = hn[-2:]
                    # shape: [BS x NR, HS x 2]
                    hist = torch.cat([hn[0], hn[1]], dim=-1)

                    # shape: [BS x NR, HS]
                    hist = self.hist_linear(hist)

                    # shape: [BS, NR, HS]
                    hist = hist.view(bs, num_rounds, self.hidden_size)

                    if self.config['model']['txt_has_pos_embedding']:
                        hist = hist + self.pos_embedding(hist)

                    if self.config['model']['txt_has_layer_norm']:
                        hist = self.layer_norm(hist)

                # shape: [BS, NH, T, hidden_size]
                hist = hist[:, None, :, :].repeat(1, num_hist, 1, 1)

                # shape: [BS, NH, T, hidden_size]
                hist = hist.masked_fill(round_mask == 0, 0.0)

                # shape: [BS x NH, T, hidden_size]
                hist = hist.view(bs * num_hist, num_rounds, self.hidden_size)

            return (hist,  # shape: [BS x NH, T, hidden_size]
                    hist_mask.long(),  # shape [BS x NH, T]
                    )
