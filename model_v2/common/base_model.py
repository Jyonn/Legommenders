from torch import nn

from model_v2.common.base_config import BaseConfig


class BaseOperator(nn.Module):
    config_class = BaseConfig

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config  # type: BaseConfig

    def forward(self, embeddings, mask=None, **kwargs):
        raise NotImplementedError
