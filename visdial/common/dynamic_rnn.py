import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DynamicRNN(nn.Module):
    """
    The wrapper version of recurrent modules including RNN, LSTM
    that support packed sequence batch.
    """

    def __init__(self, rnn_module):
        super().__init__()

        if isinstance(rnn_module, nn.LSTM):
            self.bidirectional = rnn_module.bidirectional

        self.rnn_module = rnn_module

    def forward(self, x, len_x, initial_state=None):
        """
        Arguments
        ---------
        x: torch.FloatTensor
            padded input sequence tensor for RNN model
            Shape [batch_size, max_seq_len, embed_size]
        len_x: torch.LongTensor
            Length of sequences (b, )
        initial_state: torch.FloatTensor
            Initial (hidden, cell) states of RNN model.
        Returns
        -------
        A tuple of (padded_output, h_n) or (padded_output, (h_n, c_n))
            padded_output: torch.FloatTensor
                The output of all hidden for each elements.
                Shape [batch_size, max_seq_len, hidden_size]
            h_n: torch.FloatTensor
                The hidden state of the last step for each packed sequence (not including padding elements)
                Shape [batch_size, hidden_size]
            c_n: torch.FloatTensor
                If rnn_model is RNN, c_n = None
                The cell state of the last step for each packed sequence (not including padding elements)
                Shape [batch_size, hidden_size]
        """

        # First sort the sequences in batch in the descending order of length
        sorted_len, idx = len_x.sort(dim=0, descending=True)
        sorted_x = x[idx]

        # Convert to packed sequence batch
        packed_x = pack_padded_sequence(sorted_x, lengths=sorted_len, batch_first=True)

        # Check init_state
        if initial_state is not None:
            if isinstance(initial_state, tuple):  # (h_0, c_0) in LSTM
                hx = [state[:, idx] for state in initial_state]
            else:
                hx = initial_state[:, idx]  # h_0 in RNN
        else:
            hx = None

        # Do forward pass
        self.rnn_module.flatten_parameters()
        packed_output, last_s = self.rnn_module(packed_x, hx)

        # Pad the packed_output
        max_seq_len = x.size(1)
        padded_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=max_seq_len)

        # Reverse to the original order
        _, reverse_idx = idx.sort(dim=0, descending=False)

        # shape: [BS, PaddedSEQ, HS]
        padded_output = padded_output[reverse_idx]

        if isinstance(self.rnn_module, nn.RNN):
            h_n, c_n = last_s[:, reverse_idx], None
        else:
            # shape: [num_layers x 2, BS, HS] if bidirectional
            # shape: [num_layers, BS, HS] if None
            h_n, c_n = [s[:, reverse_idx] for s in last_s]

        # The hidden cells of last layer is (h_n, h_n_inverse) is h_n[-2:, :, ]
        return padded_output, (h_n, c_n)
