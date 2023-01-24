"""
https://github.com/aqweteddy/NRMS-Pytorch/
"""

import torch
from torch import nn
from torch.nn import functional as F

from typing import Tuple, Optional


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
