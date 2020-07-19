import torch
import torch.nn as nn
from visdial.common import DynamicRNN


class GenerativeDecoder(nn.Module):
    """This Generative Decoder learn to predict ground-truth answer word-by-word during training
    and assign log-likelihood scores to all answer options (candidate answers) during evaluation.
    Given `context_vec` and and the sequences of candidate options.
    """

    def __init__(self, config):
        super(GenerativeDecoder, self).__init__()
        self.config = config
        self.test_mode = self.config['model']['test_mode']
        self.text_embedding = nn.Embedding(config['model']['txt_vocab_size'],
                                           config['model']['txt_embedding_size'],
                                           padding_idx=0)

        self.answer_lstm = nn.LSTM(
            config['model']['txt_embedding_size'],
            config['model']['hidden_size'],
            num_layers=2,
            batch_first=True,
            dropout=config['model']['dropout']
        )

        if config['model']['txt_has_decoder_layer_norm']:
            self.layer_norm = nn.LayerNorm(config['model']['hidden_size'])

        self.lstm_to_words = nn.Linear(
            config['model']['hidden_size'], config['model']['txt_vocab_size']
        )

        # self.dropout = nn.Dropout(p=config['model']['dropout'])
        self.logsoftmax = nn.LogSoftmax(dim=-1)

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
            output['opts_out_scores']: torch.FloatTensor
                The output from Generative Decoder (test mode or validation mode)
                Shape: [batch_size, NH, num_options]

            output['ans_out_scores']: torch.FloatTensor
                The output from Generative Decoder (training mode)
                Shape: Shape [batch_size, N, vocab_size]
        """
        self.answer_lstm.flatten_parameters()

        if self.training:
            # shape: [BS, NH, SEQ]
            ans_in = batch["ans_in"]
            (BS, NH, SEQ), HS = ans_in.size(), self.config['model']['hidden_size']

            # shape: [BS x NH, SEQ]
            ans_in = ans_in.view(BS * NH, SEQ)

            # shape: [BS x NH, SEQ, WE]
            ans_in_embed = self.text_embedding(ans_in)

            # reshape encoder output to be set as initial hidden state of LSTM.
            # shape: [lstm_layers, BS x NH, HS]
            num_lstm_layers = 2
            init_hidden = context_vec.view(1, BS * NH, -1).repeat(num_lstm_layers, 1, 1)

            init_cell = torch.zeros_like(init_hidden)

            # shape: [BS x NH, SEQ, HS]
            ans_out, (_, _) = self.answer_lstm(ans_in_embed, (init_hidden, init_cell))
            # ans_out = self.dropout(ans_out)

            # shape: [BS, NH, SEQ, VC]
            return {'ans_out_scores': self.lstm_to_words(ans_out).view(BS, NH, SEQ, -1)}

        else:
            opts_in = batch["opts_in"]
            target_opts_out = batch["opts_out"]

            if self.test_mode or test_mode:
                # shape: [BS, NH, NO, SEQ]
                opts_in = opts_in[:, batch['num_rounds'] - 1]
                # shape: [BS x NH x NO, SEQ]
                target_opts_out = batch["opts_out"][:, batch['num_rounds'] - 1]

            BS, NH, NO, SEQ = opts_in.size()

            target_opts_out = target_opts_out.view(BS * NH * NO, -1)

            # shape: [BS x NH x NO, SEQ]
            opts_in = opts_in.view(BS * NH * NO, SEQ)

            # shape: [BS x NH x NO, WE]
            opts_in_embed = self.text_embedding(opts_in)

            # reshape encoder output to be set as initial hidden state of LSTM.
            # shape: [BS, NH, 1, HS]
            init_hidden = context_vec.view(BS, NH, 1, -1)

            # shape: [BS, NH, NO, HS]
            init_hidden = init_hidden.repeat(1, 1, NO, 1)

            # shape: [1, BS x NH x NO, HS]
            init_hidden = init_hidden.view(1, BS * NH * NO, -1)

            num_lstm_layers = 2
            # shape: [lstm_layers, BS x NH x NO, HS]
            init_hidden = init_hidden.repeat(num_lstm_layers, 1, 1)

            init_cell = torch.zeros_like(init_hidden)

            # shape: [BS x NH x NO, SEQ, HS]
            opts_out, (_, _) = self.answer_lstm(opts_in_embed, (init_hidden, init_cell))

            if self.config['model']['txt_has_decoder_layer_norm']:
                opts_out = self.layer_norm(opts_out)

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

            return {'opts_out_scores': opts_out_scores}
