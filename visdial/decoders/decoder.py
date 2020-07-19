import torch
import torch.nn as nn
from visdial.common import SelfAttention


class Decoder(nn.Module):
    """The wrapper Decoder which includes Discriminative or Generative decoders
    """

    def __init__(self, config, disc_decoder=None, gen_decoder=None):
        super(Decoder, self).__init__()
        self.disc_decoder = disc_decoder
        self.gen_decoder = gen_decoder
        self.config = config
        hidden_size = self.config['model']['hidden_size']

        if self.config['model'].get('encoder_out') is not None:
            num_feats = len(self.config['model']['encoder_out'])

            if 'img' in self.config['model']['encoder_out']:
                self.img_summary = SelfAttention(hidden_size)

            if 'hist' in self.config['model']['encoder_out']:
                self.hist_summary = SelfAttention(hidden_size)

            if 'ques' in self.config['model']['encoder_out']:
                self.ques_summary = SelfAttention(hidden_size)

        self.context_linear = nn.Sequential(
            nn.Linear(hidden_size * num_feats, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size))

    def forward(self, batch, encoder_output, test_mode=False):
        """
        Arguments
        ---------
        batch: Dictionary 
            This provides a dictionary of inputs.
        encoder_output: A tuple of encoder output:
            img: torch.FloatTensor
                Shape [batch_size x NH, K, hidden_size]
            ques: torch.FloatTensor
                Shape [batch_size x NH, N, hidden_size]
            hist: torch.FloatTensor
                Shape [batch_size x NH, T, hidden_size]
            img_mask: torch.LongTensor
                Shape [batch_size x NH, K]
            ques_mask: torch.LongTensor
                Shape [batch_size x NH, N]
            hist_mask: torch.LongTensor
                Shape [batch_size x NH, T]

        test_mode: Boolean
            Whether the forward is performed on test data
        Returns
        -------
        output : Dictionary
            output['opt_scores']: torch.FloatTensor
                The output from Discriminative Decoder
                Shape: [batch_size, NH, num_options]
            
            output['opts_out_scores']: torch.FloatTensor
                The output from Generative Decoder (test mode or validation mode)
                Shape: [batch_size, NH, num_options]
            
            output['ans_out_scores']: torch.FloatTensor
                The output from Generative Decoder (training mode)
                Shape: Shape [batch_size, N, vocab_size]
        """
        img, ques, hist, img_mask, ques_mask, hist_mask = encoder_output

        BS, NH = batch['ques_len'].shape
        if self.config['model']['test_mode'] or test_mode:
            NH = 1

        # Perform self-attention on each utility
        encoder_output = []
        if self.config['model'].get('encoder_out') is not None:
            if 'img' in self.config['model']['encoder_out']:
                encoder_output.append(self.img_summary(img, img_mask))

            if 'hist' in self.config['model']['encoder_out']:
                encoder_output.append(self.hist_summary(hist, hist_mask))

            if 'ques' in self.config['model']['encoder_out']:
                encoder_output.append(self.ques_summary(ques, ques_mask))
            encoder_output = torch.cat(encoder_output, dim=-1)

        # shape [BS x NH, HS]
        context_vec = self.context_linear(encoder_output)
        # shape [BS, NH, HS]
        context_vec = context_vec.view(BS, NH, -1)

        output = {}
        if self.disc_decoder is not None:
            output['opt_scores'] = self.disc_decoder(batch,
                                                     context_vec,
                                                     test_mode=test_mode)['opt_scores']
        if self.gen_decoder is not None:
            if self.training:
                output['ans_out_scores'] = self.gen_decoder(batch,
                                                            context_vec,
                                                            test_mode=test_mode)['ans_out_scores']
            else:
                output['opts_out_scores'] = self.gen_decoder(batch,
                                                             context_vec,
                                                             test_mode=test_mode)['opts_out_scores']
        return output
