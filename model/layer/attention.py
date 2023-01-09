"""
https://github.com/aqweteddy/NRMS-Pytorch/
"""

import torch
from torch import nn
from torch.nn import functional as F

from typing import Tuple, Optional


class Projector(nn.Module):
    def __init__(self, hidden_size):
        super(Projector, self).__init__()

        self.hidden_size = hidden_size
        self.en_q = nn.Linear(hidden_size, hidden_size)
        self.en_k = nn.Linear(hidden_size, hidden_size)
        self.en_v = nn.Linear(hidden_size, hidden_size)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.en_q(q), self.en_k(k), self.en_v(v)


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

        attention = self.encoder(inputs).squeeze(-1)  # [B, L]
        if attention_mask is None:
            attention = torch.exp(attention)  # [B, L]
        else:
            attention = torch.exp(attention) * attention_mask  # [B, L]
        attention_weight = attention / (torch.sum(attention, dim=-1, keepdim=True) + torch.finfo(torch.float32).eps)  # [B, L]

        return torch.sum(inputs * attention_weight.unsqueeze(-1), dim=1)  # [B, D]


class ScaledDotProduct(nn.Module):

    def __init__(self, dropout=0.0):
        r"""Processes a projected query and key-value pair to apply
        scaled dot product attention.
        Args:
            dropout (float): probability of dropping an attention weight.
        Examples::
            >>> SDP = ScaledDotProduct(0.1)
            >>> q = torch.randn(256, 21, 3)
            >>> k = v = torch.randn(256, 21, 3)
            >>> attn_output, attn_weights = SDP(q, k, v)
            >>> print(attn_output.shape, attn_weights.shape)
            torch.Size([256, 21, 3]) torch.Size([256, 21, 21])
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def add_bias(x, x_bias):
        assert x.size(-1) == x_bias.size(-1) and \
               x.size(-2) == x_bias.size(-2) and \
               x_bias.size(-3) == 1, \
            "Shape of bias is not supported"
        return torch.cat([x, x_bias])

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            k_bias: Optional[torch.Tensor] = None,
            v_bias: Optional[torch.Tensor] = None,
    ) -> [torch.Tensor, torch.Tensor]:
        """
        @param q: [T, B * H, D / H]
        @param k: [S, B * H, D / H]
        @param v: [S, B * H, D / H]
        @param attention_mask: [B * H, T, M]
        @param k_bias: [1, B * H, D / H]
        @param v_bias: [1, B * H, D / H]
        @return: [T, B * H, D / H]

        T: TargetLen, S: SourceLen, B: BatchSize, H: HeadSize, D: HiddenSize
        """

        assert k_bias ^ v_bias, ValueError('k_bias and v_bias should be both exist or none')
        if k_bias and v_bias:
            k = self.add_bias(k, k_bias)
            v = self.add_bias(v, v_bias)
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, 1))

        target_len, head_dim = q.size(-3), q.size(-1)
        assert q.size(-1) == k.size(-1) == v.size(-1), "The feature dim of query, key, value must be equal."
        assert k.size() == v.size(), "Shape of key, value must match"
        source_len = k.size(-3)
        batch_heads = max(q.size(-2), k.size(-2))

        # Scale query
        q, k, v = q.transpose(-2, -3), k.transpose(-2, -3), v.transpose(-2, -3)
        q = q * (head_dim ** -0.5)
        if attention_mask:
            if attention_mask.dim() != 3:
                raise RuntimeError('attn_mask must be a 3D tensor.')
            if (attention_mask.size(-1) != source_len) or (attention_mask.size(-2) != target_len) or \
                    (attention_mask.size(-3) != 1 and attention_mask.size(-3) != batch_heads):
                raise RuntimeError('The size of the attn_mask is not correct.')
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()

        # dot product of q, k
        weights = torch.matmul(q, k.transpose(-2, -1))
        if attention_mask is not None:
            weights.masked_fill_(attention_mask, -1e8)
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        attention_output = torch.matmul(weights, v)
        return attention_output.transpose(-2, -3), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size):
        r""" A multi-head attention container
        Args:
            num_attention_heads: the number of attention heads
            hidden_size: embedding dim
        Examples::
            >>> import torch
            >>> hidden_size, num_attention_heads, batch_size = 10, 5, 64
            >>> MHA = MultiHeadAttention(num_attention_heads, hidden_size)
            >>> query = torch.rand((21, batch_size, hidden_size))
            >>> key = value = torch.rand((16, batch_size, hidden_size))
            >>> attn_output, attn_weights = MHA(query, key, value)
            >>> print(attn_output.shape)
            >>> torch.Size([21, 64, 10])
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        if self.hidden_size % self.num_attention_heads:
            raise ValueError('hidden size should be divisible by the number of attention heads')

        self.projector = Projector(hidden_size=hidden_size)
        self.attention = ScaledDotProduct()
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            k_bias: Optional[torch.Tensor] = None,
            v_bias: Optional[torch.Tensor] = None
    ) -> [torch.Tensor, torch.Tensor]:
        """

        @param q: [T, B, D]
        @param k: [S, B, D]
        @param v: [S, B, D]
        @param attention_mask: for scaled dot product attention
        @param k_bias: for scaled dot product attention
        @param v_bias: for scaled dot product attention
        @return: [T, B, D]
        """
        target_len, source_len, batch_size, hidden_size = q.size(-3), k.size(-3), q.size(-2), q.size(-1)
        q, k, v = self.projector(q, k, v)
        head_dim = hidden_size // self.num_attention_heads
        q = q.reshape(target_len, batch_size * self.num_attention_heads, head_dim)

        head_dim = hidden_size // self.num_attention_heads
        k = k.reshape(source_len, batch_size * self.num_attention_heads, head_dim)

        head_dim = hidden_size // self.num_attention_heads
        v = v.reshape(source_len, batch_size * self.num_attention_heads, head_dim)

        output, weights = self.attention(q, k, v, attn_mask=attention_mask, k_bias=k_bias, v_bias=v_bias)
        output = output.reshape(target_len, batch_size, hidden_size)
        output = self.linear(output)
        return output, weights
