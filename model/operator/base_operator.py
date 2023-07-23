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
            embed_hidden_size,
            inputer_config=None,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.embed_hidden_size = embed_hidden_size
        self.inputer_config = inputer_config or {}


class BaseOperator(nn.Module):
    config_class = BaseOperatorConfig
    inputer_class: Type[BaseInputer]
    inputer: BaseInputer

    def __init__(self, config: BaseOperatorConfig, nrd: NRDepot, embedding_manager: EmbeddingManager, target_user=False):
        super().__init__()
        self.print = printer[(self.__class__.__name__, '|', Color.GREEN)]

        self.config = config
        self.inputer = self.inputer_class(
            nrd=nrd,
            embedding_manager=embedding_manager,
            **config.inputer_config,
        )

        self.target_user = target_user

    def _get_attr_parameters(self, attr_name):
        names, params = [], []

        for name, param in self.named_parameters():
            if name.startswith(attr_name + '.') and param.requires_grad:
                names.append(name)
                params.append(param)

        return names, params

    def _get_pretrained_parameters(self):
        return [], []

    def get_pretrained_parameters(self, prefix: str):
        names, parameters = self._get_pretrained_parameters()
        names = {f'{prefix}.{name}' for name in names}
        return names, parameters

    def forward(self, embeddings, mask=None, **kwargs):
        raise NotImplementedError
