import torch
import torch.nn as nn

class SummaryAttention(nn.Module):

    def __init__(self, hidden_size):
        super(SummaryAttention, self).__init__()
        self.attn_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )
        self.attn_weights = None

    def forward(self, x, mask_x):
        # x: shape [bs, M, hidden_size]
        # mask_x: shape [bs, M]

        # shape [bs, M, 1]
        attn_weights = self.attn_linear(x)
        attn_weights = attn_weights.masked_fill(mask_x.unsqueeze(-1) == 0, value=-9e10)
        attn_weights = torch.softmax(attn_weights, dim=-2)
        self.attn_weights = attn_weights

        # shape [bs, 1, hidden_size]
        out = torch.matmul(attn_weights.transpose(-2, -1), x)
        out = out.squeeze(dim=-2)
        return out