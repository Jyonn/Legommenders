"""
https://github.com/aqweteddy/NRMS-Pytorch/
"""
from typing import cast

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    def __init__(self, embed_dim, hidden_size):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1, bias=False),
        )

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor = None) -> [torch.Tensor, torch.Tensor]:
        """

        @param inputs: [B, L, D]
        @param attention_mask: [B, L]
        @return: [B, D]
        """

        # attention = self.encoder(inputs).squeeze(-1)  # [B, L]
        # if attention_mask is None:
        #     attention = torch.exp(attention)  # [B, L]
        # else:
        #     attention = torch.exp(attention) * attention_mask  # [B, L]
        # attention_weight = attention / (torch.sum(attention, dim=-1, keepdim=True) + torch.finfo(torch.float32).eps)  # [B, L]
        #
        # return torch.sum(inputs * attention_weight.unsqueeze(-1), dim=1)  # [B, D]

        attention = self.encoder(inputs).squeeze(-1)  # [B, L]

        if attention_mask is None:
            attention = torch.exp(attention)  # [B, L]
        else:
            attention = torch.exp(attention) * attention_mask  # [B, L]

        sum_attention = torch.sum(attention, dim=-1, keepdim=True)  # [B, 1]

        # 检查是否某些样本的 attention_mask 全为 0
        mask_all_zero = cast(torch.Tensor, (sum_attention == 0)).squeeze(-1)  # [B]

        # 计算 attention_weight，避免除 0
        attention_weight = attention / (sum_attention + torch.finfo(torch.float32).eps)  # [B, L]

        # 如果某个样本的 attention_mask 全为 0，给它一个特殊默认值（如均匀分布或全 0）
        if mask_all_zero.any():
            attention_weight[mask_all_zero] = 0  # 设为全 0，确保不会影响输入

        # 计算最终的 attention 加权和
        output = torch.sum(inputs * attention_weight.unsqueeze(-1), dim=1)  # [B, D]

        return output


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention
        Ref: https://zhuanlan.zhihu.com/p/47812375
    """
    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, Q, K, V, scale=None, mask=None):
        # mask: 0 for masked positions
        scores = torch.matmul(Q, K.transpose(-1, -2))
        if scale:
            scores = scores / scale
        if mask is not None:
            mask = mask.view_as(scores)
            scores = scores.masked_fill_(mask.float() == 0, -1.e9) # fill -inf if mask=0
        attention = scores.softmax(dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.matmul(attention, V)
        return output, attention


class MultiHeadSelfAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0.,
                 use_residual=True, use_scale=False, layer_norm=False):
        super(MultiHeadSelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
            "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        if self.use_residual and input_dim != attention_dim:
            self.W_res = nn.Linear(input_dim, attention_dim, bias=False)
        else:
            self.W_res = None
        self.dot_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(attention_dim) if layer_norm else None

    def forward(self, X):
        residual = X

        # linear projection
        query = self.W_q(X)
        key = self.W_k(X)
        value = self.W_v(X)

        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot product attention
        output, attention = self.dot_attention(query, key, value, scale=self.scale)
        # concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        if self.W_res is not None:
            residual = self.W_res(residual)
        if self.use_residual:
            output += residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu()
        return output
