import torch
from torch import nn


class LRLayer(nn.Module):
    def __init__(self, use_bias=True):
        super(LRLayer, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True) if use_bias else None

    def forward(self, inputs):
        output = inputs.sum(dim=1)
        if self.bias is not None:
            output += self.bias
        return output
