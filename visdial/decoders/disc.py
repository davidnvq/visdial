import torch
import torch.nn as nn
from visdial.common.summary import SummaryAttention

from visdial.common.transformer.text_encoder import TransformerEncoder
from visdial.common.dynamic_rnn import DynamicRNN

class OptEncoder(nn.Module):

    def __init__(self, text_encoder, hidden_size):
        super(OptEncoder, self).__init__()
        self.encoder = text_encoder
        self.hidden_size = hidden_size
        if isinstance(self.encoder, TransformerEncoder):
            self.summary_linear = SummaryAttention(hidden_size)

        if isinstance(self.encoder, DynamicRNN):
            if self.encoder.bidirectional:
                self.option_linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, opt_embeds, opts_len):
        """"""

        if isinstance(self.encoder, TransformerEncoder):
            bs, num_hist, num_opts, max_seq_len, hidden_size = opt_embeds.size()

            opts = opt_embeds.view(bs * num_hist * num_opts, max_seq_len, hidden_size)
            opts_len = opts_len.view(bs * num_hist * num_opts)

            opts_mask = torch.arange(max_seq_len, device=opts.device).repeat(bs * num_hist * num_opts, 1)
            # shape[bs * num_hist * num_opts, max_seq_len]
            opts_mask = opts_mask < opts_len.unsqueeze(-1)

            # shape [bs * num_hist * num_opts, max_seq_len, hidden_size]
            opts = self.encoder(opts, opts_mask)

            # shape [bs * num_hist * num_opts, hidden_size]
            opts = self.summary_linear(opts, opts_mask)

            # shape [bs * num_hist, num_opts, hidden_size]
            return opts.view(bs * num_hist, num_opts, hidden_size)

        if isinstance(self.encoder, DynamicRNN):
            bs, num_hist, num_opts, max_seq_len, embedding_size = opt_embeds.size()

            opts = opt_embeds.view(bs * num_hist * num_opts, max_seq_len, embedding_size)
            opts_len = opts_len.view(bs * num_hist * num_opts)

            opts_mask = torch.arange(max_seq_len, device=opts.device).repeat(bs * num_hist * num_opts, 1)
            # shape[bs * num_hist * num_opts, max_seq_len]
            opts_mask = opts_mask < opts_len.unsqueeze(-1)

            # shape [bs * num_hist * num_opts, max_seq_len, hidden_size]
            _, (opts, _) = self.encoder(opts, opts_len)

            # shape [bs * num_hist * num_opts, hidden_size]
            if self.encoder.bidirectional:
                opts = opts[-2:]
                opts = torch.cat([opts[0], opts[1]], dim=-1)
                opts = self.option_linear(opts)
            else:
                opts = opts[-1]

            # shape [bs * num_hist, num_opts, hidden_size]
            return opts.view(bs * num_hist, num_opts, self.hidden_size)


class DiscDecoder(nn.Module):

    def __init__(self, opt_encoder):
        super(DiscDecoder, self).__init__()
        self.opt_encoder = opt_encoder

    def forward(self, opt_embeds, opts_len, encoder_out):
        """
        :param opt_embeds: shape [bs, num_hist, num_opts, max_seq_len, embedding_size]
        :param opts_len:   shape [bs, num_hist, num_opts]
        :param encoder_out:  shape [bs, num_hist, hidden_size]
        :return:
        """
        bs, num_hist, num_opts, _, _ = opt_embeds.size()

        # shape [bs * num_hist, num_opts, hidden_size]
        opts = self.opt_encoder(opt_embeds, opts_len)

        # shape [bs * num_hist, hidden_size, 1]
        encoder_out = encoder_out.view(bs * num_hist, -1)
        encoder_out = encoder_out.unsqueeze(-1)

        # shape [bs * num_hist, num_opts]
        opt_scores = torch.matmul(opts, encoder_out).squeeze(-1)

        # shape [bs, num_hist, num_opts]
        return opt_scores
