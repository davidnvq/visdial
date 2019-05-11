import torch
from torch import nn

def update_weights(net, path):
    pretrained_dict = torch.load(path)['model']
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)


class EncoderDecoderModel(nn.Module):
    """Convenience wrapper module, wrapping Encoder and Decoder modules.

    Parameters
    ----------
    encoder: nn.Module
    decoder: nn.Module
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch, debug=False):
        if not debug:
            encoder_output = self.encoder(batch)
            decoder_output = self.decoder(encoder_output, batch)
            return decoder_output
        else:
            encoder_output, attn_weights = self.encoder(batch, debug=debug)
            decoder_output = self.decoder(encoder_output, batch)
            return decoder_output, attn_weights