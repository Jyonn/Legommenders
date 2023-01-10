from typing import Type

from oba import Obj
from torch import nn

from model_v2.common.base_model import BaseOperator
from model_v2.inputer.base_inputer import BaseInputer
from model_v2.utils.nr_depot import NRDepot
from utils.printer import printer, Color


class BaseEncoderModel(nn.Module):
    operator_class = None  # type: Type[BaseOperator]
    inputer_class = None  # type: Type[BaseInputer]

    def __init__(self, config, nrd: NRDepot):
        super().__init__()

        self.print = printer[(self.__class__.__name__, '|', Color.GREEN)]

        operator_config = self.operator_class.config_class(**Obj.raw(config))
        self.operator = self.operator_class(config=operator_config)
        self.inputer = self.inputer_class(nrd=nrd, **Obj.raw(config.inputer_config))

    def forward(self, embeddings, mask=None, **kwargs):
        return self.operator(embeddings, mask, **kwargs)
