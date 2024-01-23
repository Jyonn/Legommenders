from typing import Type

import torch
from torch import nn

from model.common.base_module import BaseModule
from model.inputer.base_inputer import BaseInputer
from loader.embedding.embedding_hub import EmbeddingHub
from loader.data_hub import DataHub


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


class BaseOperator(BaseModule):
    config_class = BaseOperatorConfig
    inputer_class: Type[BaseInputer]
    inputer: BaseInputer
    allow_caching = True
    flatten_mode = False

    def __init__(self, config: BaseOperatorConfig, hub: DataHub, embedding_manager: EmbeddingHub, target_user=False):
        super().__init__()
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

    def get_full_placeholder(self, sample_size):
        return torch.zeros(sample_size, self.config.hidden_size, dtype=torch.float)

    # def get_full_item_placeholder(self, sample_size):
    #     return torch.zeros(sample_size, self.config.hidden_size, dtype=torch.float)

    def prepare_for_predictor(self, user_embeddings, candidate_size):
        assert self.target_user, 'repeat is only designed for user encoder'
        user_embeddings = user_embeddings.unsqueeze(1).repeat(1, candidate_size, 1)  # B, K+1, D
        user_embeddings = user_embeddings.view(-1, user_embeddings.shape[-1])
        return user_embeddings
