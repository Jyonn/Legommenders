from typing import Type

from oba import Obj
from torch import nn

from model_v2.operator.base_operator import BaseOperator
from model_v2.utils.embedding_manager import EmbeddingManager
from model_v2.utils.nr_depot import NRDepot
from utils.printer import printer, Color


class BaseEncoderModel(nn.Module):
    operator_class = None  # type: Type[BaseOperator]
    # inputer_class = None  # type: Type[BaseInputer]

    def __init__(self, config, nrd: NRDepot, embedding_manager: EmbeddingManager):
        super().__init__()

        self.config = config
        self.print = printer[(self.__class__.__name__, '|', Color.GREEN)]

        operator_config = self.operator_class.config_class(**Obj.raw(config))
        self.operator = self.operator_class(config=operator_config)
        self.inputer = self.operator_class.inputer_class(
            nrd=nrd,
            embedding_manager=embedding_manager,
            **Obj.raw(config.inputer_config)
        )

    def forward(self, embeddings, mask=None, **kwargs):
        return self.operator(embeddings, mask, **kwargs)
