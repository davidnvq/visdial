import torch
import torch.nn as nn

import logging
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
except (ImportError, AttributeError) as e:
    logging.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    LayerNorm = torch.nn.LayerNorm

from visdial.common import clones, SummaryAttention, DynamicRNN

class Encoder(nn.Module):

    def __init__(self, config, text_encoder, img_encoder, attn_encoder):
        super(Encoder, self).__init__()
        self.text_encoder = text_encoder
        self.img_encoder = img_encoder
        self.attn_encoder = attn_encoder
        self.config = config
        hidden_size = config['model']['hidden_size']

        if self.config['model'].get('encoder_out') is not None:
            num_feats = len(self.config['model']['encoder_out'])

            if 'img' in self.config['model']['encoder_out']:
                self.img_summary = SummaryAttention(hidden_size)
            if 'hist' in self.config['model']['encoder_out']:
                self.hist_summary = SummaryAttention(hidden_size)

            if 'ques' in self.config['model']['encoder_out']:
                self.ques_summary = SummaryAttention(hidden_size)

        else:
            num_feats = 2
            self.summaries = clones(SummaryAttention(hidden_size), num_feats)

        self.encoder_linear = nn.Sequential(
            nn.Linear(hidden_size * num_feats, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            LayerNorm(hidden_size))

    def forward(self, batch, test_mode=False):

        BS, NH = batch['ques_len'].shape
        if self.config['model']['test_mode'] or test_mode:
            NH = 1

        # [BS x NH, NR, HS] hist
        # [BS x NH, SQ, HS] ques
        # [BS x NH, NR] hist_mask
        # [BS x NR, SQ] ques_mask
        hist, ques, hist_mask, ques_mask = self.text_encoder(batch, test_mode=test_mode)

        # [BS x NH, NP, HS] img
        # [BS x NH, NP] img_mask
        img, img_mask = self.img_encoder(batch, test_mode=test_mode)

        batch_input = img, ques, hist, img_mask, ques_mask, hist_mask
        batch_output = self.attn_encoder(batch_input)

        img, ques, hist, img_mask, ques_mask, hist_mask = batch_output

        if self.config['model'].get('encoder_out') is not None:
            encoder_out = []

            if 'img' in self.config['model']['encoder_out']:
                encoder_out.append(self.img_summary(img, img_mask))
            if 'hist' in self.config['model']['encoder_out']:
                encoder_out.append(self.hist_summary(hist, hist_mask))
            if 'ques' in self.config['model']['encoder_out']:
                encoder_out.append(self.ques_summary(ques, ques_mask))
            encoder_out = torch.cat(encoder_out, dim=-1)

        else:
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