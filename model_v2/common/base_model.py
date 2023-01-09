from abc import ABC
from typing import Dict

from torch import nn

from model_v2.common.base_config import BaseConfig


class BaseBatch:
    def __init__(self, inputs: Dict[str, any]):
        self.inputs = inputs


class BaseOperator(nn.Module):
    config_class = BaseConfig
    batcher = BaseBatch

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config  # type: BaseConfig
