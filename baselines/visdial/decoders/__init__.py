from visdial.decoders.disc import DiscriminativeDecoder
from visdial.decoders.gen import GenerativeDecoder


def Decoder(model_config, *args):
    name_dec_map = {"disc": DiscriminativeDecoder, "gen": GenerativeDecoder}
    return name_dec_map[model_config["decoder"]](model_config, *args)
