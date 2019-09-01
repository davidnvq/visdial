import torch
from torch import nn


class ImageEncoder(nn.Module):

    def __init__(self, config):
        super(ImageEncoder, self).__init__()

        self.config = config

        self.img_linear = nn.Sequential(
            nn.LayerNorm(config['model']['img_feat_size']),
            nn.Linear(config['model']['img_feat_size'],
                      config['model']['hidden_size']),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config['model']['dropout']),
            nn.LayerNorm(config['model']['hidden_size']),
        )

        if self.config['model']['img_has_classes'] or \
                self.config['model']['img_has_attributes'] or \
                self.config['model']['img_has_bboxes']:
            self.img_norm = nn.LayerNorm(config['model']['hidden_size'])

        self.text_embedding = nn.Embedding(config['model']['txt_vocab_size'],
                                           config['model']['txt_embedding_size'])

        if self.config['model']['img_has_classes']:
            self.cls_linear = nn.Sequential(
                nn.Linear(config['model']['txt_embedding_size'],
                          config['model']['hidden_size']),
                nn.ReLU(inplace=True),
                nn.Dropout(p=config['model']['dropout']),
                nn.LayerNorm(config['model']['hidden_size'])
            )

        if self.config['model']['img_has_attributes']:
            self.attr_linear = nn.Sequential(
                nn.Linear(config['model']['txt_embedding_size'],
                          config['model']['hidden_size']),
                nn.ReLU(inplace=True),
                nn.Dropout(p=config['model']['dropout']),
                nn.LayerNorm(config['model']['hidden_size'])
            )

        if self.config['model']['img_has_bboxes']:
            self.x1_embedding = nn.Embedding(600, config['model']['hidden_size'])
            self.y1_embedding = nn.Embedding(600, config['model']['hidden_size'])

            self.x2_embedding = nn.Embedding(600, config['model']['hidden_size'])
            self.y2_embedding = nn.Embedding(600, config['model']['hidden_size'])

            self.bbox_linear = nn.Sequential(
                nn.Linear(config['model']['hidden_size'],
                          config['model']['hidden_size']),
                nn.ReLU(inplace=True),
                nn.Dropout(p=config['model']['dropout']),
                nn.LayerNorm(config['model']['hidden_size'])
            )

    def forward(self, batch):
        bs, num_hist, _ = batch['ques_tokens'].size()
        hidden_size = self.config['model']['hidden_size']

        if self.config['model']['test_mode']:
            num_hist = 1

        # shape: [batch_size, num_proposals, img_feat_size]
        img_feat = batch['img_feat']

        # img_feat: shape [bs, num_proposals, hidden_size]
        img_feat = self.img_linear(img_feat)

        # shape [bs * num_hist, num_proposals, hidden_size]
        img_feat = img_feat.unsqueeze(1).repeat(1, num_hist, 1, 1)
        img_feat = img_feat.view(bs * num_hist, -1, img_feat.size(-1))

        if batch.get('num_boxes', None) is None:
            # shape [bs * num_hist, num_proposals]
            img_mask = img_feat.new_ones(img_feat.shape[:-1], dtype=torch.long)
        else:
            # [bs,]
            num_boxes = batch['num_boxes']

            # [bs * num_hist, num_proposals]
            img_mask = torch.arange(img_feat.shape[-2], device=img_feat.device)
            img_mask = img_mask.repeat(bs * num_hist, 1)

            # [bs * num_hist, 1]
            num_boxes = num_boxes[:, None, None].repeat(1, num_hist, 1)
            num_boxes = num_boxes.view(bs * num_hist, 1)

            # [bs * num_hist, num_proposals]
            img_mask = (img_mask < num_boxes).long()

        if self.config['model']['img_has_classes']:
            # [bs, num_proposals]
            classes = batch['classes']

            # [bs * num_hist, num_proposals]
            classes = classes.unsqueeze(1).repeat(1, num_hist, 1)
            classes = classes.view(bs * num_hist, -1)

            classes = self.text_embedding(classes)
            classes = self.cls_linear(classes)
            img_feat += classes

        if self.config['model']['img_has_attributes']:
            # [bs, num_proposals, num_attrs]
            attrs = batch['attrs']

            # [bs, num_proposals, num_attrs, hidden_size]
            attrs = self.text_embedding(attrs)
            attrs = self.attr_linear(attrs)

            # [bs, num_proposals, num_attrs]
            attr_scores = batch['attr_scores']

            # [bs, num_proposals, hidden_size]
            attrs = torch.matmul(attr_scores.unsqueeze(-2), attrs).squeeze(-2)

            # [bs * num_hist, num_proposals, hidden_size]
            attrs = attrs.unsqueeze(1).repeat(1, num_hist, 1, 1)
            attrs = attrs.view(bs * num_hist, -1, hidden_size)

            img_feat += attrs

        if self.config['model']['img_has_bboxes']:
            # [bs, num_proposals, hidden_size]
            w = batch['img_w'].unsqueeze(-1).float()
            h = batch['img_h'].unsqueeze(-1).float()

            x1 = (self.bbox_linear(self.x1_embedding((batch['boxes'][:, :, 0] * 600 / w).long())))
            y1 = (self.bbox_linear(self.y1_embedding((batch['boxes'][:, :, 1] * 600 / h).long())))
            x2 = (self.bbox_linear(self.x2_embedding((batch['boxes'][:, :, 2] * 600 / w).long())))
            y2 = (self.bbox_linear(self.y2_embedding((batch['boxes'][:, :, 3] * 600 / h).long())))

            # [bs, num_proposals, hidden_size]
            bboxes = (x1 + y1 + x2 + y2) / 4.0

            # [bs, num_hist, num_proposals, hidden_size]
            bboxes = bboxes.unsqueeze(1).repeat(1, num_hist, 1, 1)

            # [bs, num_hist, num_proposals, hidden_size]
            bboxes = bboxes.view(bs * num_hist, -1, hidden_size)

            img_feat += bboxes

        if self.config['model']['img_has_classes'] or \
                self.config['model']['img_has_attributes'] or self.config['model']['img_has_bboxes']:
            img_feat = self.img_norm(img_feat)

        return img_feat, img_mask
