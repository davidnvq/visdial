import torch
import torch.nn as nn

from visdial.common.embeddings import TextEmbeddings
from visdial.common.dynamic_rnn import DynamicRNN
from visdial.common.summary import SummaryAttention
from visdial.common.transformer.text_encoder import TransformerEncoder


class TextEncoder(nn.Module):
    def __init__(self, text_embeddings, hist_encoder, ques_encoder):
        super(TextEncoder, self).__init__()
        self.text_embeddings = text_embeddings
        self.hist_encoder = hist_encoder
        self.ques_encoder = ques_encoder


    def forward(self, batch):

        # ques_tokens: shape [bs, num_hist, max_seq_len]
        # hist_tokens: shape [bs, num_hist, num_rounds, max_seq_len]
        ques_tokens = batch['ques_tokens']
        hist_tokens = batch['hist_tokens']

        # ques_len: shape [bs, num_hist]
        # hist_len: shape [bs, num_hist, num_rounds]
        ques_len = batch['ques_len']
        hist_len = batch['hist_len']

        # ques: shape [bs, num_hist, max_seq_len, embedding_size(or hidden_size)]
        # hist: shape [bs, num_hist, num_rounds, max_seq_len, embedding_size(or hidden_size)]
        ques = self.text_embeddings(ques_tokens)
        hist = self.text_embeddings(hist_tokens)

        # ques: shape [bs * num_hist, max_seq_len, hidden_size]
        # hist: shape [bs * num_hist, num_rounds, hidden_size]
        # ques_mask: shape [bs * num_hist, max_seq_len]
        # hist_mask: shape [bs * num_hist, num_rounds]
        ques, ques_mask = self.ques_encoder(ques, batch['ques_len'])
        hist, hist_mask = self.hist_encoder(hist, batch['hist_len'])

        return hist, ques, hist_mask, ques_mask


class QuesEncoder(nn.Module):

    def __init__(self, text_encoder, hidden_size, split='train'):
        super(QuesEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = text_encoder
        self.ques_linear = nn.Linear(hidden_size*2, hidden_size)
        self.split = split


    def forward(self, ques, ques_len):
        """
        :param ques:        shape [bs, num_hist, max_seq_len, embedding_size]
        :param ques_len:    shape [bs, num_hist]
        :return:
               ques         shape [bs, num_hist, max_seq_len, hidden_size]
               ques_mask    shape [bs, num_hist, max_seq_len]
        """
        # for test only
        if self.split == 'test':
            # get only the last question
            last_idx = (ques_len > 0).sum()
            ques = ques[:, last_idx-1:last_idx]
            ques_len = ques_len[:, last_idx-1:last_idx]

        bs, num_hist, max_seq_len, embedding_size = ques.size()

        # shape [bs * num_hist, max_seq_len, hidden_size]
        ques = ques.view(bs * num_hist, max_seq_len, embedding_size)

        # shape [bs * num_hist]
        ques_len = ques_len.view(bs * num_hist)

        # shape [bs * num_hist, max_seq_len]
        ques_mask = torch.arange(max_seq_len, device=ques.device).repeat(bs * num_hist, 1)
        ques_mask = ques_mask < ques_len.unsqueeze(-1)

        if isinstance(self.encoder, TransformerEncoder):
            # ques:         shape [bs * num_hist, max_seq_len, hidden_size]
            # ques_mask:    shape [bs * num_hist, max_seq_len,]
            # shape [bs * num_hist, max_seq_len, hidden_size]

            ques = self.encoder(ques, ques_mask)
            return ques, ques_mask.long()

        if isinstance(self.encoder, DynamicRNN):
            # LSTM
            if self.encoder.bidirectional == False:
                # shape: ques [bs * num_hist, max_seq_len, hidden_size]
                ques, (_, _) = self.encoder(ques, ques_len)

                # shape: ques [bs * num_hist, max_seq_len, hidden_size]
                # shape: ques [bs * num_hist, max_seq_len,]
                return ques, ques_mask.long()

            # BiLSTM
            else:
                # [BS x NH, SEQ, HS x 2]
                ques, (_, _) = self.encoder(ques, ques_len)

                # [BS x NH, SEQ, HS]
                ques = self.ques_linear(ques)
                return ques, ques_mask.long()


class HistEncoder(nn.Module):

    def __init__(self, text_encoder, hidden_size, split='train'):
        self.hidden_size = hidden_size
        super(HistEncoder, self).__init__()
        self.encoder = text_encoder
        self.hist_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.split=split

        if isinstance(text_encoder, TransformerEncoder):
            self.summary_linear = SummaryAttention(hidden_size)


    def forward(self, hist, hist_len):
        bs, num_rounds, max_seq_len, embedding_size = hist.size()

        # shape [bs * num_rounds, max_seq_len, hidden_size]
        hist = hist.view(bs * num_rounds, max_seq_len, embedding_size)

        # # shape [bs * num_hist * num_rounds, max_seq_len]
        # hist_mask = torch.arange(max_seq_len, device=hist.device).repeat(bs * num_hist * num_rounds, 1)
        # hist_mask = hist_mask < hist_len.unsqueeze(-1)

        # shape [bs * num_rounds]
        hist_len = hist_len.view(bs * num_rounds)

        if self.split == 'test':
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

            # shape [bs, num_hist, num_rounds, 1]
            round_mask = MASK[None, :, :, None].repeat(bs, 1, 1, 1)

            hist_mask = MASK[None, :, :].repeat(bs, 1, 1).view(bs * num_hist, num_rounds)

        if isinstance(self.encoder, DynamicRNN):

            if self.encoder.bidirectional == False: # LSTM
                # shape: [num_layers, BS, HS] if Not bidirectional
                # shape: hn = [num_layers, bs * num_rounds, hidden_size]
                y, (hn, cn) = self.encoder(hist, hist_len)

                # shape: [bs * num_rounds, hidden_size]
                hist = hn[-1]

                # shape: [bs, num_rounds, hidden_size]
                hist = hist.view(bs, num_rounds, self.hidden_size)

            else: # BiLSTM

                # hn [num_layers x 2 (bidirectional), BS x NR, HS]
                y, (hn, cn) = self.encoder(hist, hist_len)

                # ReDAN use y and softmax for each words to filtering the irrelevant words
                # hist = attn * y (where attn = softmax(W1 * tanh (W2 * y))

                # shape: [2, BS x NR, HS]
                hn = hn[-2:]
                # shape: [BS x NR, HS x 2]
                hist = torch.cat([hn[0], hn[1]], dim=-1)

                # shape: [BS x NR, HS]
                hist = self.hist_linear(hist)

                # shape [BS, NR, HS]
                hist = hist.view(bs, num_rounds, self.hidden_size)

            # shape: [bs, num_hist, num_rounds, hidden_size]
            hist = hist[:, None, :, :].repeat(1, num_hist, 1, 1)

            # shape: [bs, num_hist, num_rounds, hidden_size]
            hist = hist.masked_fill(round_mask == 0, 0.0)

            # shape: [bs * num_hist, num_rounds, hidden_size]
            hist = hist.view(bs * num_hist, num_rounds, self.hidden_size)


        return (hist, # shape: [bs * num_hist, num_rounds, hidden_size]
                hist_mask.long(), # shape [bs * num_hist, num_rounds]
                )