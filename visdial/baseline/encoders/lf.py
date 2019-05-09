import torch
from torch import nn
from torch.nn import functional as F

from visdial.utils import DynamicRNN


class LateFusionEncoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config

        self.word_embed = nn.Embedding(
            len(vocabulary),
            config["word_embedding_size"],
            padding_idx=vocabulary.PAD_INDEX,
        )
        self.hist_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )
        self.ques_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )
        self.dropout = nn.Dropout(p=config["dropout"])

        # questions and history are right padded sequences of variable length
        # use the DynamicRNN utility module to handle them properly
        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.ques_rnn = DynamicRNN(self.ques_rnn)

        # project image features to lstm_hidden_size for computing attention
        self.image_features_projection = nn.Linear(
            config["img_feature_size"], config["lstm_hidden_size"]
        )

        # fc layer for image * question to attention weights
        self.attention_proj = nn.Linear(config["lstm_hidden_size"], 1)

        # fusion layer (attended_image_features + question + history)
        fusion_size = (
            config["img_feature_size"] + config["lstm_hidden_size"] * 2
        )
        self.fusion = nn.Linear(fusion_size, config["lstm_hidden_size"])

        nn.init.kaiming_uniform_(self.image_features_projection.weight)
        nn.init.constant_(self.image_features_projection.bias, 0)
        nn.init.kaiming_uniform_(self.fusion.weight)
        nn.init.constant_(self.fusion.bias, 0)

    def forward(self, batch, debug=False):

        # shape: (batch_size, img_feature_size) - CNN fc7 features
        # shape: (batch_size, num_proposals, img_feature_size) - RCNN features
        # shape: [BS, NP, IS]
        img = batch["img_feat"]

        # shape: [BS, 10, SEQ]
        ques = batch["ques"]

        # shape: [BS, 10, SEQ x 2 x 10] <- concatenated q & a * 10 rounds
        hist = batch["hist"]

        # num_rounds = 10, even for test (padded dialog rounds at the end)
        (BS, NR, SEQ), HS = ques.size(), self.config["lstm_hidden_size"]
        NP, IS = img.size(1), img.size(2)

        # embed questions
        # shape: [BS x NR, SEQ]
        ques = ques.view(BS * NR, SEQ)

        # shape: [BS x NR, SEQ, WE]
        ques_embed = self.word_embed(ques)

        # shape: [BS x NR, HS]
        _, (ques_embed, _) = self.ques_rnn(ques_embed, batch["ques_len"])

        # embed history
        # shape: [BS x NR, SEQ x 20]
        hist = hist.view(BS * NR, SEQ * 20)

        # shape: [BS x NR, SEQ x 20, WE]
        hist_embed = self.word_embed(hist)

        # shape: [BS x NR, HS]
        _, (hist_embed, _) = self.hist_rnn(hist_embed, batch["hist_len"])

        # project down image features and ready for attention
        # shape: [BS, NP, HS]
        projected_image_features = self.image_features_projection(img)

        # TODONE: below lines are the same as baseline

        # shape: [BS, 1, NP, HS]
        projected_image_features = projected_image_features.view(BS, 1, -1, HS)

        # shape: [BS, NR, 1, HS]
        projected_ques_features = ques_embed.view(BS, NR, 1, HS)

        # shape: [BS, NR, NP, HS]
        projected_ques_image = projected_image_features * projected_ques_features
        projected_ques_image = self.dropout(projected_ques_image)

        # computing attention weights
        # shape: [BS, NR, NP, 1]
        image_attention_weights = self.attention_proj(projected_ques_image)

        # shape: [BS, NR, NP, 1]
        image_attention_weights = F.softmax(image_attention_weights, dim=-2) # <- dim = NP

        # shape: [BS, 1, NP, IS]
        img = img.view(BS, 1, NP, IS)

        # shape: [BS, NR, NP, 1] * [BS, (1), NP, IS] -> [BS, NR, NP, IS]
        attended_image_features = image_attention_weights * img

        # shape: [BS, NR, IS]
        img = attended_image_features.sum(dim=-2) # dim=NP

        # shape: [BS x NR, IS]
        img = img.view(BS * NR, IS)

        # shape: [BS x NR, IS + HSx2]
        fused_vector = torch.cat((img, ques_embed, hist_embed), 1)
        fused_vector = self.dropout(fused_vector)

        # shape: [BS x NR, HS]
        fused_embedding = torch.tanh(self.fusion(fused_vector))

        # shape: [BS, NR, HS]
        fused_embedding = fused_embedding.view(BS, NR, -1)

        if debug:
            return fused_embedding, image_attention_weights.squeeze(-1)
        return fused_embedding
