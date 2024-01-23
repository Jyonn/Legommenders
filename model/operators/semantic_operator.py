from typing import Type

import torch
from torch import nn

from model.common.attention import AdditiveAttention
from model.inputer.semantic_inputer import SemanticInputer
from model.operators.base_operator import BaseOperator, BaseOperatorConfig
from utils.function import combine_config


class SemanticOperatorConfig(BaseOperatorConfig):
    def __init__(
            self,
            user_operator: str,
            user_operator_config: dict,
            additive_hidden_size: int = 256,
            **kwargs
    ):
        super().__init__(**kwargs)

        from loader.class_hub import ClassHub
        operators = ClassHub.operators()
        self.user_operator_class = operators(user_operator)  # type: Type[BaseOperator]
        self.user_operator_config = self.user_operator_class.config_class(
            **combine_config(config=user_operator_config, **kwargs)
        )

        self.additive_hidden_size = additive_hidden_size


class SemanticOperator(BaseOperator):
    config_class = SemanticOperatorConfig
    config: SemanticOperatorConfig
    inputer_class = SemanticInputer
    inputer: SemanticInputer
    flatten_mode = True

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)

        assert self.target_user, 'semantic operator is only designed as user encoder'

        self.num_semantic_layers = self.inputer.get_max_content_len()

        user_operators = []
        for i in range(self.num_semantic_layers):
            base_operator = self.config.user_operator_class(config=self.config.user_operator_config, **kwargs)
            user_operators.append(base_operator)
        self.user_operators = nn.ModuleList(user_operators)

        self.additive_attention = AdditiveAttention(
            embed_dim=self.config.hidden_size,
            hidden_size=self.config.additive_hidden_size,
        )

    def forward(self, embeddings, mask=None, **kwargs):
        """
        @param embeddings: [B, L, S, D]
        @param mask: [B, L]
        @return: [B, D]
        """
        # [B, L, S, D] to [S, B, L, D]
        embeddings = embeddings.transpose(0, 2)  # [S, B, L, D]
        embeddings = embeddings.transpose(1, 2)  # [S, L, B, D]
        increment = torch.zeros(embeddings.shape[1:]).to(embeddings.device)  # [B, L, D]
        user_embeddings = []

        for i in range(self.num_semantic_layers):
            operator = self.user_operators[i]
            increment = increment + embeddings[i]
            user_embeddings.append(operator(increment, mask=mask))  # [B, D]

        user_embeddings = torch.stack(user_embeddings, dim=1)  # [B, S, D]
        user_embeddings = self.additive_attention(user_embeddings)  # [B, D]
        return user_embeddings

    # def prepare_for_predictor(self, user_embeddings, candidate_size):
    #     assert self.target_user, 'repeat is only designed for user encoder'
    #     # user_embeddings: [B, S, D] to [B, S, C, D]
    #     user_embeddings = user_embeddings.unsqueeze(2).repeat(1, 1, candidate_size, 1)
    #     # transpose to [S, B, C, D]
    #     user_embeddings = user_embeddings.transpose(0, 1)
    #     user_embeddings = user_embeddings.reshape(user_embeddings.shape[0], -1, user_embeddings.shape[-1])  # [S, B*C, D]
    #     return user_embeddings
