from typing import Type

import torch
from torch import nn

from model.inputer.base_inputer import BaseInputer
from loader.embedding.embedding_hub import EmbeddingHub
from loader.data_hub import DataHub
from utils.printer import printer, Color


class BaseOperatorConfig:
    def __init__(
            self,
            hidden_size,
            embed_hidden_size,
            input_dim,
            inputer_config=None,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.embed_hidden_size = embed_hidden_size
        self.input_dim = input_dim
        self.inputer_config = inputer_config or {}


class BaseOperator(nn.Module):
    config_class = BaseOperatorConfig
    inputer_class: Type[BaseInputer]
    inputer: BaseInputer

    def __init__(self, config: BaseOperatorConfig, hub: DataHub, embedding_manager: EmbeddingHub, target_user=False):
        super().__init__()
        self.print = printer[(self.__class__.__name__, '|', Color.GREEN)]

        self.config = config
        self.inputer = self.inputer_class(
            hub=hub,
            embedding_manager=embedding_manager,
            **config.inputer_config,
        )

        self.target_user = target_user

    def get_pretrained_parameter_names(self):
        return []

    def forward(self, embeddings, mask=None, **kwargs):
        raise NotImplementedError

    def get_full_item_placeholder(self, sample_size):
        return torch.zeros(sample_size, self.config.hidden_size, dtype=torch.float)
