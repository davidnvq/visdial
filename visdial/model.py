import os
import torch
from torch import nn
from visdial.decoders import DiscriminativeDecoder, GenerativeDecoder, Decoder
from visdial.encoders import ImageEncoder, TextEncoder, HistEncoder, QuesEncoder, AttentionStackEncoder, Encoder


class VisdialModel(nn.Module):
    def __init__(self, encoder, decoder, init_type='kaiming_uniform'):
        super(VisdialModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.init_type = init_type
        self.apply(self.weight_init)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)

    def forward(self, batch, test_mode=False):
        return self.decoder(batch, self.encoder(batch, test_mode=test_mode), test_mode=test_mode)

    def weight_init(self, m):
        init_dict = {
            'kaiming_uniform': nn.init.kaiming_uniform_,
            'kaiming_normal': nn.init.kaiming_normal_,
        }

        if isinstance(m, nn.Linear):
            init_dict[self.init_type](m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def get_attn_encoder(config):
    encoder = Encoder(
        config=config,
        text_encoder=TextEncoder(config, HistEncoder(config), QuesEncoder(config)),
        img_encoder=ImageEncoder(config),
        attn_encoder=AttentionStackEncoder(config),
    )
    return encoder


def get_disc_model(config):
    encoder = get_attn_encoder(config)
    encoder.img_encoder.text_embedding = encoder.text_encoder.text_embedding

    disc_decoder = DiscriminativeDecoder(config)
    disc_decoder.text_embedding = encoder.text_encoder.text_embedding

    gen_decoder = None
    decoder = Decoder(config, disc_decoder, gen_decoder)

    model = VisdialModel(encoder, decoder)
    return model


def get_gen_model(config):
    encoder = get_attn_encoder(config)
    encoder.img_encoder.text_embedding = encoder.text_encoder.text_embedding

    disc_decoder = None
    gen_decoder = GenerativeDecoder(config)
    gen_decoder.text_embedding = encoder.text_encoder.text_embedding
    decoder = Decoder(config, disc_decoder, gen_decoder)
    return VisdialModel(encoder, decoder)


def get_misc_model(config):
    encoder = get_attn_encoder(config)
    encoder.img_encoder.text_embedding = encoder.text_encoder.text_embedding

    disc_decoder = DiscriminativeDecoder(config)
    disc_decoder.text_embedding = encoder.text_encoder.text_embedding

    gen_decoder = GenerativeDecoder(config)
    gen_decoder.text_embedding = encoder.text_encoder.text_embedding

    decoder = Decoder(config, disc_decoder, gen_decoder)
    return VisdialModel(encoder, decoder)


def get_model(config):
    get_model_dict = {
        'gen': get_gen_model,
        'disc': get_disc_model,
        'misc': get_misc_model
    }

    model = get_model_dict[config['model']['decoder_type']](config)
    glove_path = os.path.expanduser(config['dataset']['glove_path'])
    glove_weights = torch.load(glove_path)
    model.encoder.text_encoder.text_embedding.load_state_dict(glove_weights)
    return model
