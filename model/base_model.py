from torch import nn


class BaseConfig:
    def __init__(self, **kwargs):
        pass


class BaseModel(nn.Module):
    configer = BaseConfig

    def __init__(self, **kwargs):
        super().__init__()
        self.config = self.configer(**kwargs)
