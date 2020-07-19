import torch
import torch.nn as nn
from visdial.common import DynamicRNN


class DiscriminativeDecoder(nn.Module):
    """The Discriminative Decoder computes the rankings (prediction scores) for each option (candidate answers)
    Given `context_vec` and sequences candidate options.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.test_mode = self.config['model']['test_mode']

        self.text_embedding = nn.Embedding(config['model']['txt_vocab_size'],
                                           config['model']['txt_embedding_size'],
                                           padding_idx=0)

        self.opt_lstm = nn.LSTM(config['model']['txt_embedding_size'],
                                config['model']['hidden_size'],
                                num_layers=2,
                                batch_first=True,
                                dropout=config['model']['dropout'],
                                bidirectional=config['model']['txt_bidirectional'])

        if config['model']['txt_has_decoder_layer_norm']:
            self.layer_norm = nn.LayerNorm(config['model']['hidden_size'])

        self.option_linear = nn.Linear(config['model']['hidden_size'] * 2,
                                       config['model']['hidden_size'])

        # Options are variable length padded sequences, use DynamicRNN.
        self.opt_lstm = DynamicRNN(self.opt_lstm)

    def forward(self, batch, context_vec, test_mode=False):
        """
        Arguments
        ---------
        batch: Dictionary
            This provides a dictionary of inputs.
        context_vec: torch.FloatTensor
            The context vector summarized from all utilities
            Shape [batch_size, NH, hidden_size]

        test_mode: Boolean
            Whether the forward is performed on test data
        Returns
        -------
        output : Dictionary
            output['opt_scores']: torch.FloatTensor
                The output from Discriminative Decoder
                Shape: [batch_size, NH, num_options]
        """
        # shape: [BS, NH, NO, SEQ]
        options = batch["opts"]

        # batch_size, num_rounds, num_opts, seq_len
        BS, NH, NO, SEQ = options.size()
        HS = self.config['model']['hidden_size']

        # shape: [BS x NH x NO, SEQ]
        options = options.view(BS * NH * NO, SEQ)

        # shape: [BS, NH, NO]
        options_length = batch["opts_len"]

        # shape: [BS x NH x NO]
        options_length = options_length.view(BS * NH * NO)

        # Pick options with non-zero length (relevant for test split).
        # shape: [BS x (nR x NO)] <- nR ~= 1 or 10 for test: nR = 1, for train, val nR = 10
        nonzero_options_length_indices = options_length.nonzero().squeeze()

        # shape: [BS x (nR x NO)]
        nonzero_options_length = options_length[nonzero_options_length_indices]

        # shape: [BS x (nR x NO)]
        nonzero_options = options[nonzero_options_length_indices]

        # shape: [BS x NH x NO, SEQ, WE]
        # shape: [BS x 1  x NO, SEQ, WE] <- FOR TEST SPLIT
        nonzero_options_embed = self.text_embedding(nonzero_options)

        # shape: [lstm_layers x bi, BS x NH x NO, HS]
        # shape: [lstm_layers x bi, BS x 1  x NO, HS] FOR TEST SPLIT,
        _, (nonzero_options_embed, _) = self.opt_lstm(nonzero_options_embed, nonzero_options_length)

        # shape: [2, BS x NH x NO, HS]
        nonzero_options_embed = nonzero_options_embed[-2:]

        # shape: [BS x NH x NO, HS x 2]
        nonzero_options_embed = torch.cat([nonzero_options_embed[0], nonzero_options_embed[1]], dim=-1)

        # shape: [BS x NH x NO, HS]
        nonzero_options_embed = self.option_linear(nonzero_options_embed)

        # shape: [BS x NH x NO, HS] <- move back to standard for TEST split
        options_embed = torch.zeros(BS * NH * NO, HS, device=options.device)

        if self.config['model']['txt_has_decoder_layer_norm']:
            options_embed = self.layer_norm(options_embed)

        # shape: [BS x NH x NO, HS]
        options_embed[nonzero_options_length_indices] = nonzero_options_embed

        # shape: [BS, NH, HS] -> [BS, NH, HS, 1]
        context_vec = context_vec.unsqueeze(-1)
        # shape: [BS, NH, NO, HS]
        options_embed = options_embed.view(BS, NH, NO, -1)

        # shape: [BS, NH, NO, 1]
        scores = torch.matmul(options_embed, context_vec)

        # shape: [BS, NH, NO]
        scores = scores.squeeze(-1)

        if self.test_mode:
            scores = scores[:, batch['num_rounds'] - 1]

        return {'opt_scores': scores}
