''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

# from transformer.Modules import ScaledDotProductAttention

__author__ = "yakunLi"


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=2048, scale_emb=False):
        super().__init__()

        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx, _weight=init_emd)
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, input_q, input_k, input_v, src_mask, return_attns=False, add_positon_emd=True):

        enc_slf_attn_list = []

        # -- Forward
        # enc_output = self.src_word_emb(src_seq)
        enc_output = input_q
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        if add_positon_emd:
            enc_output = self.position_enc(enc_output)
            input_k = self.position_enc(input_k)
            input_v = self.position_enc(input_v)

        enc_output = self.layer_norm(self.dropout(enc_output))
        input_k = self.layer_norm(self.dropout(input_k))
        input_v = self.layer_norm(self.dropout(input_v))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, input_k, input_v, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_q, enc_k, enc_v, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_q, enc_k, enc_v, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

def nn_init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def norm_init(module, weight_init, bias_init, mean=0, std=1.0):
    weight_init(module.weight.data, mean=mean, std=std)
    bias_init(module.bias.data)
    return module

def commmon_init(module, weight_init, bias_init):
    weight_init(module.weight.data)
    bias_init(module.bias.data)
    return module


def nn_init_method(init_type='norm'):
    if init_type == 'none':
        init_method = lambda x: x
    elif init_type == 'norm':
        init_method = lambda m: norm_init(m, nn.init.normal_, lambda x: nn.init.constant_(x, 0))
    elif init_type == 'kaiming_normal':
        init_method = lambda m: commmon_init(m, nn.init.kaiming_normal_, lambda x: nn.init.constant_(x, 0))
    elif init_type == 'xavier_normal':
        init_method = lambda m: nn_init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0))
    else:
        init_method = lambda m: nn_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
    return init_method


def neural_norm(hidden_size, norm_type="bn"):
    if norm_type == 'bn':
        norm = nn.BatchNorm1d(hidden_size)
    elif norm_type == 'ln':
        norm = nn.LayerNorm(hidden_size)
    elif norm_type == 'gn':
        norm = group_norm(hidden_size)
    else:
        norm = lambda x: x
    return norm

def neural_dropout(dropout_ratio):
    if dropout_ratio == 0:
        # dropout = lambda x: x
        dropout = nn.Identity()
    else:
        assert 0 < dropout_ratio < 1
        dropout = nn.Dropout(dropout_ratio)

    return dropout


def neural_nonlinear(activation="relu"):
    if activation == 'relu':
        nonlinear = nn.ReLU()
    elif activation == 'elu':
        nonlinear = nn.ELU()
    elif activation == 'leaky_relu':
        nonlinear = nn.LeakyReLU()
    elif activation == 'gelu':
        nonlinear = nn.GELU()
    return nonlinear


class affinityReadout(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(self, hidden_size, dim_per_head=4, n_head=None, d_inner=None, n_layers=1, pad_idx=0, dropout_rate=0.1, n_position=2048, scale_emb=False, mlp_layer_num=1):
        super().__init__()
        d_model = hidden_size
        if d_inner is None:
            d_inner = d_model * 4
        self.hidden_size = hidden_size
        d_k = d_v = dim_per_head
        n_head = d_model // dim_per_head
        init_ = nn_init_method()
        self.ab_ag_encoder = Encoder(n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_k,
            d_model=d_model, d_inner=d_inner, pad_idx=pad_idx, dropout=dropout_rate, n_position=n_position, scale_emb=scale_emb)

        self.ag_ab_encoder = Encoder(n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_k,
            d_model=d_model, d_inner=d_inner, pad_idx=pad_idx, dropout=dropout_rate, n_position=n_position, scale_emb=scale_emb)

        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.mlp_layer_num = mlp_layer_num
        self.out_projection = init_(nn.Linear(hidden_size * 2, hidden_size))
        self.fc_list = nn.ModuleList([init_(nn.Linear(hidden_size, hidden_size)) for _ in range(mlp_layer_num)])
        self.norm_list = nn.ModuleList([neural_norm(hidden_size) for _ in range(mlp_layer_num)])
        self.dropout_list = nn.ModuleList([neural_dropout(dropout_rate) for _ in range(mlp_layer_num)])
        self.nonlinear = neural_nonlinear()
        self.out_linear = init_(nn.Linear(hidden_size, 1))

    def forward(self, ab_feature, ag_feature):
        outputs = {}
        ab_ag_feature = self.ab_ag_encoder(ab_feature, ag_feature, ag_feature, None, add_positon_emd=False)[
            -1]  # [B*, Lh, h]
        ag_ab_feature = self.ag_ab_encoder(ag_feature, ab_feature, ab_feature, None, add_positon_emd=False)[
            -1]  # [B*, Lh, h]

        ab_ag_emd = torch.squeeze(self.pooling(ab_ag_feature.transpose(1, 2)),
                                   dim=-1)  # (b, hidden_dim, l1) -> (b, hidden_dim, 1)

        ag_ab_emd = torch.squeeze(self.pooling(ag_ab_feature.transpose(1, 2)),
                                   dim=-1)  # (b, hidden_dim, l1) -> (b, hidden_dim, 1)

        emd = torch.cat([ab_ag_emd, ag_ab_emd], dim=-1)  # (b, hidden_dim*2)
        emd = self.out_projection(emd)

        for i in range(self.mlp_layer_num):
            # (b, hidden_dim) -> (b, hidden_dim)
            emd = self.nonlinear(self.dropout_list[i](self.norm_list[i](self.fc_list[i](emd)))) + emd
        preds = self.out_linear(self.nonlinear(emd))  # (b, hidden_dim) -> (b, 1)
        outputs['preds'] = preds
        return outputs