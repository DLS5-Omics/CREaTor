import torch
from torch import nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, q, k, v, bias=None):
        attn = torch.matmul(q / self.scale, k.transpose(-1, -2))
        if bias is not None:
            attn += bias
        attn_m = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn_m, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        if d_model % n_head != 0:
            raise ValueError("The hidden size is not a multiple of the number of attention heads")

        self.n_head = n_head
        self.d_k = d_model // n_head
        self.fc_query = nn.Linear(d_model, d_model, bias=False)
        self.fc_key = nn.Linear(d_model, d_model, bias=False)
        self.fc_value = nn.Linear(d_model, d_model, bias=False)
        self.attention = ScaledDotProductAttention(scale=self.d_k ** 0.5, dropout=dropout)
        self.fc_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout, inplace=False)

    def transpose_for_scores(self, x):
        """
        x has shape (*, L, C)
        return shape (*, nhead, L, C/nhead)
        """
        new_shape = x.shape[:-1] + (self.n_head, -1)
        x = x.view(*new_shape)
        return x.transpose(-3, -2)

    def forward(self, x, bias=None):
        q = self.transpose_for_scores(self.fc_query(x))
        k = self.transpose_for_scores(self.fc_key(x))
        v = self.transpose_for_scores(self.fc_value(x))
        x, attn = self.attention(q, k, v, bias=bias)
        x = x.transpose(-3, -2)
        x = x.reshape(*x.shape[:-2], -1)
        x = self.dropout(self.fc_out(x))
        return x, attn


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout, inplace=False),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout, inplace=False),
        )

    def forward(self, x):
        return self.ff(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model=d_model, n_head=n_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout)

    def forward(self, x, bias):
        branch, attn = self.attn(self.norm1(x), bias)
        x = x + branch
        x = x + self.ffn(self.norm2(x))
        return x, attn


class TransformerEncoder(nn.Module):
    def __init__(self, n_layer, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(**kwargs) for _ in range(n_layer)])

    def forward(self, x, bias):
        attn_weight = []
        for module in self.layers:
            x, w = module(x, bias)
            attn_weight.append(w)
        return x, attn_weight
