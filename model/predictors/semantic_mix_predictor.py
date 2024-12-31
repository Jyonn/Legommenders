from typing import Type

import torch
from torch import nn

from loader.env import Env
from model.predictors.base_predictor import BasePredictor, BasePredictorConfig
from utils.function import combine_config


class SemanticMixPredictorConfig(BasePredictorConfig):
    def __init__(
            self,
            # num_layers,
            base_predictor: str = 'DCN',
            base_predictor_config: dict = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # self.num_layers = num_layers
        base_predictor_config = base_predictor_config or {}

        from loader.class_hub import ClassHub
        predictors = ClassHub.predictors()
        self.base_predictor_class = predictors(base_predictor)  # type: Type[BasePredictor]

        self.base_predictor_config = self.base_predictor_class.config_class(
            **combine_config(config=base_predictor_config, **kwargs)
        )
        # self.base_predictor = base_predictor_class(config=self.base_predictor_config)  # type: BasePredictor


class SemanticMixPredictor(BasePredictor):
    config_class = SemanticMixPredictorConfig
    config: SemanticMixPredictorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_item_semantics = self.lego_config.item_ut.cols[self.lego_config.item_inputs[0]].max_length
        self.num_user_semantics = self.lego_config.user_ut.cols[self.lego_config.user_inputs[0]].max_length
        self.linear = nn.Linear(self.num_item_semantics * self.num_user_semantics, 1)

        self.masks = []
        self.base_predictor = self.config.base_predictor_class(
            config=self.config.base_predictor_config,
            lego_config=self.lego_config,
        )

    def build_base_predictor(self):
        return self.config.base_predictor_class(
            config=self.config.base_predictor_config,
            lego_config=self.lego_config,
        )

    @staticmethod
    def get_empty_placeholder(embeddings):
        shape = list(embeddings.shape)
        shape[1] = shape[1] * 2 - 1
        return torch.zeros(shape, dtype=torch.float).to(Env.device)

    def predict(self, user_embeddings, item_embeddings):
        """
        @param user_embeddings: [B, Su, D]   batch size, user semantics, embedding size
        @param item_embeddings: [B, Si, D]   batch size, item semantics, embedding size
        @return:
        """

        batch_size = user_embeddings.shape[0]

        for i in range(1, user_embeddings.shape[1]):
            user_embeddings[:, i, :] = user_embeddings[:, i, :] + user_embeddings[:, i - 1, :]
        for i in range(1, item_embeddings.shape[1]):
            item_embeddings[:, i, :] = item_embeddings[:, i, :] + item_embeddings[:, i - 1, :]

        user_embeddings = user_embeddings.unsqueeze(1)  # [B, 1, Su, D]
        item_embeddings = item_embeddings.unsqueeze(2)  # [B, Si, 1, D]
        user_embeddings = user_embeddings.repeat(1, item_embeddings.size(1), 1, 1)  # [B, Si, Su, D]
        item_embeddings = item_embeddings.repeat(1, 1, user_embeddings.size(2), 1)  # [B, Si, Su, D]

        user_embeddings = user_embeddings.view(-1, user_embeddings.size(-1))  # [B*Si*Su, D]
        item_embeddings = item_embeddings.view(-1, item_embeddings.size(-1))  # [B*Si*Su, D]

        scores = self.base_predictor(user_embeddings, item_embeddings)  # [B * Si * Su]
        scores = scores.reshape(batch_size, -1)  # [B, Si * Su]
        scores = self.linear(scores)  # [B, 1]

        return scores.flatten()
