import torch
import torch.nn as nn
from visdial.common.utils import clones, check_flag


class NormalSubLayer(nn.Module):
    """Perform Linear Projection with Dropout and RelU activation inside the for all MHAttn"""

    def __init__(self, hidden_size, dropout):
        super(NormalSubLayer, self).__init__()
        self.linear = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=dropout))

    def forward(self, x):
        """x: shape [batch_size, M, hidden_size*3]"""
        return self.linear(x)


class MultiHeadAttention(nn.Module):
    """This module perform MultiHeadAttention for 2 utilities X, and Y as follows:
    MHA_Y(X) = MHA(X, Y, Y) and
    MHA_X(Y) = MHA(Y, X, X).
    This can be done with sharing similarity matrix since
        X_query = X_key = X_value
        Y_query = Y_key = Y_value
        Then sim_matrix(X_query, Y_key) = sim_matrix(Y_query, X_key)
    Please refer to our paper and supplementary for more details.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.hidden_size = config['model']['hidden_size']
        self.num_heads = config['model']['ca_num_attn_heads']
        self.pad_size = config['model']['ca_pad_size']
        self.d_h = self.hidden_size // self.num_heads

        self.pad_x = torch.empty(self.pad_size, self.hidden_size)
        self.pad_x = nn.Parameter(nn.init.kaiming_uniform_(self.pad_x))
        self.pad_y = torch.empty(self.pad_size, self.hidden_size)
        self.pad_y = nn.Parameter(nn.init.kaiming_uniform_(self.pad_y))

        self.attn_X_guided_by_Y = None
        self.attn_Y_guided_by_X = None

    def project(self, X, pad_x):
        """
        Project X into X_query, X_key, X_value (all are X_proj) by
        splitting along last indexes mechanically.
        Note that: X_query = X_key = X_value = X_proj since W_Q = W_K = W_V.
        Arguments
        ---------
        X: torch.FloatTensor
            The input tensor with
            Shape [batch_size, M, hidden_size]
        pad_x: torch.FloatTensor
            The padding vectors we would like to put at the beginning of X
            Shape [batch_size, pad_size, hidden_size]
        Returns
        -------
        X_proj: torch.FloatTensor
            The summarized vector of the utility (the context vector for this utility)
            Shape [batch_size, M + pad_size, num_heads, d_h]
        """
        size = X.size(0), self.pad_size, self.hidden_size
        X = torch.cat([pad_x.unsqueeze(0).expand(*size), X], dim=1)

        X_proj = X.view(X.size(0), X.size(1), self.num_heads, self.d_h)
        return X_proj

    def forward(self, X, Y, mask_X, mask_Y):
        """
        Arguments
        ---------
        X: torch.FloatTensor
            The input tensor of utility X
            Shape [batch_size, M, hidden_size]
        Y: torch.FloatTensor
            The input tensor of utility Y
            Shape [batch_size, N, hidden_size]
        mask_X: torch.LongTensor
            The mask of utility X where 0 denotes <PAD>
            Shape [batch_size, M]
        mask_Y: torch.LongTensor
            The mask of utility Y where 0 denotes <PAD>
            Shape [batch_size, N]
        Returns
        -------
        A tuple of two MultiHeadAttention
            A_X(Y): torch.FloatTensor
                The attention from the source Y to X: Y_attends_in_X
                Shape [batch_size, M, hidden_size]
            A_Y(X): torch.FloatTensor
                The attention from the source X to Y: X_attends_in_Y
                Shape [batch_size, N, hidden_size]
        """
        pad_mask = X.new_ones((X.size(0), self.pad_size)).long()

        mask_X = torch.cat([pad_mask, mask_X], dim=1)
        mask_Y = torch.cat([pad_mask, mask_Y], dim=1)
        M_pad, N_pad = mask_X.size(1), mask_Y.size(1)
        mask_X = mask_X[:, None, :, None].repeat(1, self.num_heads, 1, N_pad)
        mask_Y = mask_Y[:, None, None, :].repeat(1, self.num_heads, M_pad, 1)

        # X_proj: [bs, pad_size + M, num_heads, d_h]
        X_proj = self.project(X, self.pad_x)

        # Y_proj [bs, pad_size + N, num_heads, d_h]
        Y_proj = self.project(Y, self.pad_y)

        # (1) shape [bs, num_heads, pad_size + M, d_h]
        # (2) shape [bs, num_heads, d_h, pad_size + N]
        X_proj = X_proj.permute(0, 2, 1, 3)
        Y_proj = Y_proj.permute(0, 2, 3, 1)

        """
        Note that:
        X_query = X_key = X_value = X_proj,
        Y_query = Y_key = Y_value = Y_proj
        Then, we have sim_matrix(X_query, Y_key) = sim_matrix(Y_query, X_key) = sim_matrix
        """
        # shape: [bs, num_heads, pad_size + M, pad_size + N]
        sim_matrix = torch.matmul(X_proj, Y_proj)
        sim_matrix = sim_matrix.masked_fill(mask_X == 0, -1e9)
        sim_matrix = sim_matrix.masked_fill(mask_Y == 0, -1e9)

        # shape: [bs, num_heads, pad_size + M, pad_size + N]
        attn_X_guided_by_Y = torch.softmax(sim_matrix, dim=2)
        attn_Y_guided_by_X = torch.softmax(sim_matrix, dim=3)

        # shape [bs, num_heads, pad_size + M, d_h]
        X_value = X_proj
        # shape [bs, num_heads, pad_size + N, d_h]
        X_attends_in_Y = torch.matmul(attn_X_guided_by_Y.transpose(2, 3), X_value)
        # shape [bs, num_heads, N, d_h]
        X_attends_in_Y = X_attends_in_Y[:, :, self.pad_size:, :]
        # shape [bs, N, num_heads, d_h]
        X_attends_in_Y = X_attends_in_Y.permute(0, 2, 1, 3).contiguous()
        # shape [bs, N, num_heads, hidden_size]
        X_attends_in_Y = X_attends_in_Y.view(X_attends_in_Y.size(0), X_attends_in_Y.size(1), -1)

        # shape [bs, num_heads, pad_size + N, d_h]
        Y_value = Y_proj.permute(0, 1, 3, 2).contiguous()
        # shape [bs, num_heads, pad_size + M, d_h]
        Y_attends_in_X = torch.matmul(attn_Y_guided_by_X, Y_value)
        # shape [bs, num_heads, M, d_h]
        Y_attends_in_X = Y_attends_in_X[:, :, self.pad_size:, :]
        # shape [bs, M, num_heads, d_h]
        Y_attends_in_X = Y_attends_in_X.permute(0, 2, 1, 3).contiguous()
        # shape [bs, M, hidden_size]
        Y_attends_in_X = Y_attends_in_X.view(Y_attends_in_X.size(0), Y_attends_in_X.size(1), -1)

        # for later visualization
        if self.config['model']['ca_has_avg_attns']:
            X_value = X_value.permute(0, 2, 1, 3).contiguous()
            # shape [bs, pad_size + M, hidden_size]
            X_value = X_value.view(X_value.size(0), X_value.size(1), -1)
            Y_value = Y_value.permute(0, 2, 1, 3).contiguous()
            # shape [bs, pad_size + N, hidden_size]
            Y_value = Y_value.view(Y_value.size(0), Y_value.size(1), -1)
            attn_X_guided_by_Y = torch.mean(attn_X_guided_by_Y, dim=1)
            attn_Y_guided_by_X = torch.mean(attn_Y_guided_by_X, dim=1)
            self.attn_X_guided_by_Y = attn_X_guided_by_Y
            self.attn_Y_guided_by_X = attn_Y_guided_by_X
            # shape: [bs, pad_size + N, hidden_size]
            X_attends_in_Y = torch.matmul(attn_X_guided_by_Y.transpose(1, 2), X_value)
            # shape: [bs, pad_size + M, hidden_size]
            Y_attends_in_X = torch.matmul(attn_Y_guided_by_X, Y_value)
            # shape: [bs, N, hidden_size]
            X_attends_in_Y = X_attends_in_Y[:, self.pad_size:, :]
            # shape: [bs, M, hidden_size]
            Y_attends_in_X = Y_attends_in_X[:, self.pad_size:, :]
        return X_attends_in_Y, Y_attends_in_X


class AttentionStack(nn.Module):
    """
    The Attention Stack include of 3 blocks (i.e. 9 MHAttentions) to compute the
    attention from all sources to one target (including itself)
    Attention from X -> Y and Y -> X can be wrapped into a single MultiHeadAttention
    And self-attention X -> X: can be wrapped into MultiHeadAttention(X, X)
    """

    def __init__(self, config):
        super(AttentionStack, self).__init__()
        self.config = config
        hidden_size = config['model']['hidden_size']
        dropout = config['model']['dropout']

        self.co_attns = clones(MultiHeadAttention(config), 3)
        if check_flag(self.config['model'], 'ca_has_self_attns'):
            self.self_attns = clones(MultiHeadAttention(config), 3)

        self.im_mlp = NormalSubLayer(hidden_size, dropout)
        self.qe_mlp = NormalSubLayer(hidden_size, dropout)
        self.hi_mlp = NormalSubLayer(hidden_size, dropout)

        if self.config['model']['ca_has_layer_norm']:
            self.im_norm = nn.LayerNorm(hidden_size)
            self.qe_norm = nn.LayerNorm(hidden_size)
            self.hi_norm = nn.LayerNorm(hidden_size)

    def forward(self, triples):
        """
        Arguments
        ---------
        triples: A tuple of the following:
            im: torch.FloatTensor
                The representation of image utility
                Shape [batch_size x NH, K, hidden_size]
            qe: torch.FloatTensor
                The representation of question utility
                Shape [batch_size x NH, N, hidden_size]
            hi: torch.FloatTensor
                The representation of history utility
                Shape [batch_size x NH, T, hidden_size]
            mask_im: torch.LongTensor
                Shape [batch_size x NH, K]
            mask_qe: torch.LongTensor
                Shape [batch_size x NH, N]
            mask_hi: torch.LongTensor
                Shape [batch_size x NH, T]
        Returns
        -------
        output : A tuples of the updated representations of inputs as the triples.
        """
        im, qe, hi, mask_im, mask_qe, mask_hi = triples
        im_in_qe, qe_in_im = self.co_attns[0](im, qe, mask_im, mask_qe)
        im_in_hi, hi_in_im = self.co_attns[1](im, hi, mask_im, mask_hi)
        qe_in_hi, hi_in_qe = self.co_attns[2](qe, hi, mask_qe, mask_hi)

        if check_flag(self.config['model'], 'ca_has_self_attns'):
            im_in_im, _ = self.self_attns[0](im, im, mask_im, mask_im)
            hi_in_hi, _ = self.self_attns[1](hi, hi, mask_hi, mask_hi)
            qe_in_qe, _ = self.self_attns[2](qe, qe, mask_qe, mask_qe)
            a_im = self.im_mlp(torch.cat([im_in_im, qe_in_im, hi_in_im], dim=-1))
            a_qe = self.qe_mlp(torch.cat([qe_in_qe, hi_in_qe, im_in_qe], dim=-1))
            a_hi = self.hi_mlp(torch.cat([hi_in_hi, qe_in_hi, im_in_hi], dim=-1))
        else:
            a_im = self.im_mlp(torch.cat([im, qe_in_im, hi_in_im], dim=-1))
            a_qe = self.qe_mlp(torch.cat([qe, hi_in_qe, im_in_qe], dim=-1))
            a_hi = self.hi_mlp(torch.cat([hi, qe_in_hi, im_in_hi], dim=-1))

        if self.config['model']['ca_has_residual']:
            im = im + a_im
            qe = qe + a_qe
            hi = hi + a_hi
        else:
            im = a_im
            qe = a_qe
            hi = a_hi

        if self.config['model']['ca_has_layer_norm']:
            im = self.im_norm(im)
            qe = self.qe_norm(qe)
            hi = self.hi_norm(hi)

        return im, qe, hi, mask_im, mask_qe, mask_hi


class AttentionStackEncoder(nn.Module):
    """
    This provide L attention stacks in the encoder
    """

    def __init__(self, config):
        super(AttentionStackEncoder, self).__init__()
        self.config = config

        num_cross_attns = self.config['model']['ca_num_attn_stacks']

        # whether to share the attention weights or not
        if self.config['model']['ca_has_shared_attns']:
            layers = [AttentionStack(config)] * num_cross_attns
        else:
            layers = [AttentionStack(config) for _ in range(num_cross_attns)]

        self.cross_attn_encoder = nn.Sequential(*layers)

    def forward(self, triples):
        return self.cross_attn_encoder(triples)
