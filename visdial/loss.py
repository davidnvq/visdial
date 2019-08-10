import torch
import torch.nn as nn


class FinetuneLoss(nn.Module):
	def __init__(self):
		super(FinetuneLoss, self).__init__()

	def forward(self, scores, batch):
		# scores [BS, NH, NO]
		BS, NH, NO = scores.size()
		relev_round_indices = batch['round_id'] - 1 # Must be -1
		# [BS, 1, NO]
		relev_round_indices = relev_round_indices[:, None, None].repeat(1, 1, NO)
		# [BS, 1, NO]
		scores = torch.gather(scores, 1, relev_round_indices)
		# [BS, NO]
		scores = scores.squeeze(dim=1)
		scores = nn.functional.log_softmax(scores, dim=-1)

		loss = torch.mean((batch['gt_relevance'] * scores)) * (-1)
		return loss