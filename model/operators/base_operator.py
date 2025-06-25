from typing import Type

import torch
from torch import nn

from model.inputer.base_inputer import BaseInputer


class BaseOperatorConfig:
    def __init__(
            self,
            hidden_size,  # dim of network hidden states
            input_dim,  # dim of input embeddings
            inputer_config=None,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.inputer_config = inputer_config or {}


class BaseOperator(nn.Module):
    config_class = BaseOperatorConfig
    inputer_class: Type[BaseInputer]
    inputer: BaseInputer
    allow_caching = True
    flatten_mode = False

    def __init__(self, config: BaseOperatorConfig, lego_config, target_user=False):
        super().__init__()
        from model.lego_config import LegoConfig

        self.config: BaseOperatorConfig = config
        self.target_user: bool = target_user
        self.lego_config: LegoConfig = lego_config

        if target_user:
            ut, inputs = lego_config.user_ut, lego_config.user_inputs
        else:
            ut, inputs = lego_config.item_ut, lego_config.item_inputs

        self.inputer = self.inputer_class(
            ut=ut,
            inputs=inputs,
            eh=self.lego_config.eh,
            **self.config.inputer_config,
        )

    def get_pretrained_parameter_names(self):
        return []

    def forward(self, embeddings, mask=None, **kwargs):
        raise NotImplementedError

    def get_full_placeholder(self, sample_size):
        return torch.zeros(sample_size, self.config.hidden_size, dtype=torch.float)

    @property
    def output_dim(self):
        return self.config.hidden_size

    # def get_full_item_placeholder(self, sample_size):
    #     return torch.zeros(sample_size, self.config.hidden_size, dtype=torch.float)

    def prepare_for_predictor(self, user_embeddings, candidate_size):
        assert self.target_user, 'repeat is only designed for user encoder'
        user_embeddings = user_embeddings.unsqueeze(1).repeat(1, candidate_size, 1)  # B, K+1, D
        user_embeddings = user_embeddings.view(-1, user_embeddings.shape[-1])
        return user_embeddings

    @property
    def classname(self):
        return self.__class__.__name__
