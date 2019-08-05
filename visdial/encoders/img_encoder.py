import torch
from torch import nn

class ImageEncoder(nn.Module):

	def __init__(self, img_feat_size, hidden_size, dropout=0.0):
		super(ImageEncoder, self).__init__()

		self.img_linear = nn.Sequential(
				nn.Linear(img_feat_size, hidden_size),
				# ReDAN use tanh here
				nn.Dropout(p=dropout)
				)
		# nn.init.kaiming_uniform_(self.img_linear[0].weight)
		# nn.init.constant_(self.img_linear[0].bias, 0)

	def forward(self, batch):
		bs, num_hist, _ = batch['ques_tokens'].size()
		# shape: [batch_size, num_proposals, img_feat_size]
		img_feat = batch['img_feat']

		# img_feat: shape [bs, num_proposals, hidden_size]
		img_feat = self.img_linear(img_feat)

		# shape [bs * num_hist, num_proposals, hidden_size]
		img_feat = img_feat.unsqueeze(1).repeat(1, num_hist, 1, 1)
		img_feat = img_feat.view(bs * num_hist, -1, img_feat.size(-1))

		# shape [bs * num_hist, num_proposals]
		img_mask = img_feat.new_ones(img_feat.shape[:-1], dtype=torch.long)

		return img_feat, img_mask
		

