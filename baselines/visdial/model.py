import torch
from torch import nn

def update_weights(net, path):
    pretrained_dict = torch.load(path)['model']
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    for key in pretrained_dict:
        special_keys = ['encoder.ques_rnn.rnn_model.weight_ih_l1',
                        'encoder.hist_rnn.rnn_model.weight_ih_l1']
        if key in special_keys:
            pretrained_dict[key] = torch.cat([pretrained_dict[key]]*2, dim=-1)
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

DIS_PATH = '/home/ubuntu/datasets/visdial/checkpoints/baselines/lf_disc_faster_rcnn_x101_trainval.pth'
GEN_PATH = '/home/ubuntu/datasets/visdial/checkpoints/baselines/lf_gen_faster_rcnn_x101_train.pth'


class EncoderDecoderModel(nn.Module):
    """Convenience wrapper module, wrapping Encoder and Decoder modules.

    Parameters
    ----------
    encoder: nn.Module
    decoder: nn.Module
    """
    def __init__(self, encoder, decoder, is_bilstm=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        if is_bilstm:
            if self.decoder.config['decoder'] == 'disc':
                update_weights(self, DIS_PATH)
            if self.decoder.config['decoder'] == 'gen':
                update_weights(self, GEN_PATH)


    def forward(self, batch, debug=False):
        if not debug:
            encoder_output = self.encoder(batch)
            decoder_output = self.decoder(encoder_output, batch)
            return decoder_output
        else:
            encoder_output, attn_weights = self.encoder(batch, debug=debug)
            decoder_output = self.decoder(encoder_output, batch)
            return decoder_output, attn_weights