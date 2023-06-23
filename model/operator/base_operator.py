from typing import Type

from torch import nn

from model.inputer.base_inputer import BaseInputer
from loader.embedding.embedding_manager import EmbeddingManager
from model.utils.nr_depot import NRDepot
from utils.printer import printer, Color


class BaseOperatorConfig:
    def __init__(
            self,
            hidden_size,
            inputer_config=None,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.inputer_config = inputer_config or {}


class BaseOperator(nn.Module):
    config_class = BaseOperatorConfig
    inputer_class: Type[BaseInputer]

    def __init__(self, config: BaseOperatorConfig, nrd: NRDepot, embedding_manager: EmbeddingManager):
        super().__init__()
        self.print = printer[(self.__class__.__name__, '|', Color.GREEN)]

        self.config = config
        self.inputer = self.inputer_class(
            nrd=nrd,
            embedding_manager=embedding_manager,
            **config.inputer_config,
        )

    def get_pretrained_parameters(self):
        return []

    def forward(self, embeddings, mask=None, **kwargs):
        raise NotImplementedError
