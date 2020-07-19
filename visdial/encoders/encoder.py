import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, config, text_encoder, img_encoder, attn_encoder):
        super(Encoder, self).__init__()
        self.text_encoder = text_encoder
        self.img_encoder = img_encoder
        self.attn_encoder = attn_encoder
        self.config = config

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
        batch_output: a tuple of the following
            im: torch.FloatTensor
                The representation of image utility
                Shape [batch_size x NH, K, hidden_size]
            qe: torch.FloatTensor
                The representation of question utility
                Shape [batch_size x NH, N, hidden_size]
            hi: torch.FloatTensor
                The representation of history utility
                Shape [batch_size x NH, T, hidden_size]
            mask_im: torch.LongTensor
                Shape [batch_size x NH, K]
            mask_qe: torch.LongTensor
                Shape [batch_size x NH, N]
            mask_hi: torch.LongTensor
                Shape [batch_size x NH, T]

        It is noted
            K is num_proposals,
            T is the number of rounds
            N is the max sequence length in the question.
        """

        # [BS x NH, T, HS] hist
        # [BS x NH, N, HS] ques
        # [BS x NH, T] hist_mask
        # [BS x NH, N] ques_mask
        hist, ques, hist_mask, ques_mask = self.text_encoder(batch, test_mode=test_mode)

        # [BS x NH, K, HS] img
        # [BS x NH, K] img_mask
        img, img_mask = self.img_encoder(batch, test_mode=test_mode)

        batch_input = img, ques, hist, img_mask, ques_mask, hist_mask
        batch_output = self.attn_encoder(batch_input)
        return batch_output
