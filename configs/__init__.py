# from .lf_disc_lstm_config import get_lf_disc_lstm_config
# from .lf_gen_lstm_config import get_lf_gen_lstm_config
# from .lf_misc_lstm_config import get_lf_misc_lstm_config
from .attn_disc_lstm_config import get_attn_disc_lstm_config
from .attn_gen_lstm_config import get_attn_gen_lstm_config
from .attn_misc_lstm_config import get_attn_misc_lstm_config

get_config_dict = {
	# 'lf_gen_lstm'   : get_lf_gen_lstm_config(),
	# 'lf_disc_lstm'  : get_lf_disc_lstm_config(),
	# 'lf_misc_lstm'  : get_lf_misc_lstm_config(),
	'attn_gen_lstm' : get_attn_gen_lstm_config(),
	'attn_disc_lstm': get_attn_disc_lstm_config(),
	'attn_misc_lstm': get_attn_misc_lstm_config()
	}


def get_config(config_name='lf_disc_lstm'):
	return get_config_dict[config_name]
