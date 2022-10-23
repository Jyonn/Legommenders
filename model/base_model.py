from torch import nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        pass
