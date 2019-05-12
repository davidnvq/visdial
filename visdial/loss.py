import torch.nn as nn


def get_criterion(decoder):
	"""get loss"""

	if decoder == "disc":
		criterion = nn.CrossEntropyLoss()
	elif decoder == "gen":
		criterion = nn.CrossEntropyLoss(ignore_index=0)
	else:
		raise NotImplementedError()

	return criterion
