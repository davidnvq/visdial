from visdial.encoders.lf import LateFusionEncoder
from visdial.encoders.attn import AttentionEncoder


def Encoder(model_config, *args):
    name_enc_map = {"lf": LateFusionEncoder, "attn" : AttentionEncoder}
    return name_enc_map[model_config["encoder"]](model_config, *args)
