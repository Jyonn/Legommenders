from abc import ABC
from typing import Type

from torch import nn

from model_v2.common.base_model import BaseOperator
from model_v2.inputer.base_inputer import BaseInputer


class BaseEncoderConfig:
    def __init__(self, inputer_config, **kwargs):
        self.inputer_config = inputer_config
        self.operator_config = kwargs


class BaseEncoderModel(nn.Module):
    config_class = None  # type: Type[BaseEncoderConfig]
    operator_class = None  # type: Type[BaseOperator]
    inputer_class = None  # type: Type[BaseInputer]

    def __init__(self, config: BaseEncoderConfig):
        super().__init__()
        self.config = config  # type: BaseEncoderConfig

        operator_config = self.operator_class.config_class(**config.operator_config)
        self.operator = self.operator_class(config=operator_config)
        self.inputer = self.inputer_class(config=config.inputer_config)

    def forward(self, embeddings, **kwargs):
        return self.operator(embeddings)
