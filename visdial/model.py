import torch
from torch import nn
from visdial.common.dynamic_rnn import DynamicRNN
from visdial.common.embeddings import TextEmbeddings
from visdial.encoders.encoder import LateFusionEncoder
from visdial.decoders.decoder import DiscriminativeDecoder, GenerativeDecoder, MiscDecoder
from visdial.encoders import ImageEncoder, TextEncoder, HistEncoder, QuesEncoder, CrossAttentionEncoder, Encoder
from visdial.decoders import Decoder, DiscDecoder, OptEncoder

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


    def forward(self, batch):
        return self.decoder(batch, self.encoder(batch))

    def weight_init(self, m):

        init_dict = {
            'kaiming_uniform': nn.init.kaiming_uniform_,
            'kaiming_normal' : nn.init.kaiming_normal_,
        }

        if isinstance(m, nn.Linear):
            init_dict[self.init_type](m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def get_attn_encoder(config):
    text_embeddings = TextEmbeddings(
        vocab_size=config['model']['vocab_size'],
        embedding_size=config['model']['embedding_size'],
        hidden_size=config['model']['hidden_size'],
        has_position=config['model']['embedding_has_position'],
        has_hidden_layer=config['model']['embedding_has_hidden_layer']
    )

    def get_lstm(config):
        lstm = DynamicRNN(nn.LSTM(config['model']['embedding_size'],
                                  config['model']['hidden_size'],
                                  num_layers=2,
                                  bidirectional=config['model']['bidirectional'],
                                  batch_first=True))
        return lstm

    encoder = Encoder(
        text_encoder=TextEncoder(
            text_embeddings,
            HistEncoder(get_lstm(config), hidden_size=config['model']['hidden_size']),
            QuesEncoder(get_lstm(config), hidden_size=config['model']['hidden_size']),
        ),

        img_encoder=ImageEncoder(
            dropout=config['model']['dropout'],
            hidden_size=config['model']['hidden_size'],
            img_feat_size=config['model']['img_feature_size']
        ),
        attn_encoder=CrossAttentionEncoder(
            hidden_size=config['model']['hidden_size'],
            num_heads=config['model']['encoder_num_heads'],
            memory_size=config['model']['encoder_memory_size'],
            num_cross_attns=config['model']['encoder_num_cross_attns']
        ),
        hidden_size=config['model']['hidden_size']
    )
    return encoder


def get_attn_disc_lstm_model(config):
    encoder = get_attn_encoder(config)
    disc_decoder = DiscriminativeDecoder(config)
    disc_decoder.word_embed = encoder.text_encoder.text_embeddings.tok_embedding

    gen_decoder = None
    decoder = MiscDecoder(disc_decoder, gen_decoder)

    model = VisdialModel(encoder, decoder)
    return model


def get_attn_gen_lstm_model(config):
    encoder = get_attn_encoder(config)

    disc_decoder = None
    gen_decoder = GenerativeDecoder(config)
    gen_decoder.word_embed = encoder.text_encoder.text_embeddings.tok_embedding

    decoder = MiscDecoder(disc_decoder, gen_decoder)

    return VisdialModel(encoder, decoder)


def get_attn_misc_lstm_model(config):
    encoder = get_attn_encoder(config)
    disc_decoder = DiscriminativeDecoder(config)
    disc_decoder.word_embed = encoder.text_encoder.text_embeddings.tok_embedding

    gen_decoder = GenerativeDecoder(config)
    gen_decoder.word_embed = encoder.text_encoder.text_embeddings.tok_embedding

    decoder = MiscDecoder(disc_decoder, gen_decoder)

    return VisdialModel(encoder, decoder)


def get_lf_disc_lstm_model(config):
    encoder = LateFusionEncoder(config)
    decoder = DiscriminativeDecoder(config)
    decoder.word_embed = encoder.word_embed

    model = VisdialModel(encoder=encoder, decoder=decoder)
    return model


def get_lf_gen_lstm_model(config):
    encoder = LateFusionEncoder(config)
    decoder = GenerativeDecoder(config)
    decoder.word_embed = encoder.word_embed

    model = VisdialModel(encoder=encoder, decoder=decoder)
    return model


def get_lf_misc_lstm_model(config):
    encoder = LateFusionEncoder(config)

    disc_decoder = DiscriminativeDecoder(config)
    disc_decoder.word_embed = encoder.word_embed

    gen_decoder = GenerativeDecoder(config)
    gen_decoder.word_embed = gen_decoder.word_embed

    decoder = MiscDecoder(disc_decoder=disc_decoder, gen_decoder=gen_decoder)
    model = VisdialModel(encoder=encoder, decoder=decoder)
    return model


def get_model(config):
    get_model_dict = {
        'lf_gen_lstm'   : get_lf_gen_lstm_model,
        'lf_disc_lstm'  : get_lf_disc_lstm_model,
        'lf_misc_lstm'  : get_lf_misc_lstm_model,
        'attn_gen_lstm' : get_attn_gen_lstm_model,
        'attn_disc_lstm': get_attn_disc_lstm_model,
        'attn_misc_lstm': get_attn_misc_lstm_model
    }

    model = get_model_dict[config['config_name']](config)
    glove_path = config['dataset']['glove']
    glove_weights = torch.load(glove_path)
    model.encoder.text_encoder.text_embeddings.tok_embedding.load_state_dict(glove_weights)
    return model
