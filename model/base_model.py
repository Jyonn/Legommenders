from torch import nn


class BaseConfig:
    def __init__(self, **kwargs):
        pass


class BaseModel(nn.Module):
    config_class = BaseConfig

    def __init__(self, **kwargs):
        super().__init__()
