from typing import Type, Dict, cast, Optional

import torch
from pigmento import pnt
from torch import nn

from loader.meta import Meta
from model.common.mediator import ModuleType
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

        self.num_item_semantics: Optional[int] = None
        self.num_user_semantics: Optional[int] = None
        self.minor_semantics: Optional[int] = None
        self.linear = None

        # self.base_predictors = nn.ModuleList()
        self.masks = []
        self.base_predictor = self.config.base_predictor_class(config=self.config.base_predictor_config)

    def build_base_predictor(self):
        return self.config.base_predictor_class(config=self.config.base_predictor_config)

    def request(self) -> Dict[str, list]:
        return {
            ModuleType.legommender: ['self']
        }

    def receive(self, responser_name: str, response: dict):
        from model.legommender import Legommender
        if responser_name == ModuleType.legommender:
            legommender = cast(Legommender, response['self'])
            self.num_item_semantics = legommender.item_hub.depot.cols[legommender.item_hub.order[0]].max_length
            self.num_user_semantics = legommender.user_hub.depot.cols[legommender.user_hub.order[0]].max_length
            self.linear = nn.Linear(self.num_item_semantics * self.num_user_semantics, 1)

            # for i in range(self.num_item_semantics):
            #     line_predictors = nn.ModuleList()
            #     for j in range(self.num_user_semantics):
            #         line_predictors.append(self.build_base_predictor())
            #     self.base_predictors.append(line_predictors)

            # self.minor_semantics = min(self.num_item_semantics, self.num_user_semantics)
            # for i in range(self.minor_semantics):
            #     self.base_predictors.append(self.build_base_predictor())
            #
            # for i in range(self.minor_semantics):
            #     mask = torch.zeros(self.minor_semantics, self.minor_semantics, dtype=torch.float).to(Meta.device)
            #     mask[i, :i + 1] = 1
            #     mask[:i + 1, i] = 1
            #     self.masks.append(mask)

    @staticmethod
    def get_empty_placeholder(embeddings):
        shape = list(embeddings.shape)
        shape[1] = shape[1] * 2 - 1
        return torch.zeros(shape, dtype=torch.float).to(Meta.device)

    def predict(self, user_embeddings, item_embeddings):
        """
        @param user_embeddings: [B, Su, D]   batch size, user semantics, embedding size
        @param item_embeddings: [B, Si, D]   batch size, item semantics, embedding size
        @return:
        """

        num_user_semantics = user_embeddings.shape[1]
        num_item_semantics = item_embeddings.shape[1]

        # _user_embeddings = self.get_empty_placeholder(user_embeddings)
        # _item_embeddings = self.get_empty_placeholder(item_embeddings)
        #
        # _user_embeddings[:, :num_user_semantics, :] = user_embeddings
        # _item_embeddings[:, :num_item_semantics, :] = item_embeddings
        #
        batch_size = user_embeddings.shape[0]
        #
        # for i in range(1, user_embeddings.shape[1]):
        #     user_embeddings[:, i, :] = user_embeddings[:, i, :] + user_embeddings[:, i - 1, :]
        # for i in range(1, item_embeddings.shape[1]):
        #     item_embeddings[:, i, :] = item_embeddings[:, i, :] + item_embeddings[:, i - 1, :]
        #
        # _user_embeddings[:, num_user_semantics:, :] = user_embeddings[:, 1:, :]
        # _item_embeddings[:, num_item_semantics:, :] = item_embeddings[:, 1:, :]
        #
        # user_embeddings = _user_embeddings
        # item_embeddings = _item_embeddings

        # print(user_embeddings.shape, item_embeddings.shape)
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

        # scores = torch.zeros(batch_size, num_item_semantics * num_user_semantics, dtype=torch.float).to(Meta.device)
        # for i in range(num_item_semantics):
        #     for j in range(num_user_semantics):
        #         user_embedding = user_embeddings[:, j, :]
        #         item_embedding = item_embeddings[:, i, :]
        #         scores[:, i * num_user_semantics + j] = self.base_predictors[i][j](user_embedding, item_embedding)
        # scores = self.linear(scores)

        # user_embeddings = user_embeddings[:, -self.minor_semantics:, :]
        # item_embeddings = item_embeddings[:, -self.minor_semantics:, :]
        #
        # user_embeddings = user_embeddings.unsqueeze(1)  # [B, 1, Su, D]
        # item_embeddings = item_embeddings.unsqueeze(2)  # [B, Si, 1, D]
        # user_embeddings = user_embeddings.repeat(1, item_embeddings.size(1), 1, 1)  # [B, Si, Su, D]
        # item_embeddings = item_embeddings.repeat(1, 1, user_embeddings.size(2), 1)  # [B, Si, Su, D]
        #
        # user_embeddings = user_embeddings.view(-1, user_embeddings.size(-1))  # [B*Si*Su, D]
        # item_embeddings = item_embeddings.view(-1, item_embeddings.size(-1))  # [B*Si*Su, D]
        #
        # scores = torch.zeros(batch_size, self.minor_semantics, self.minor_semantics, dtype=torch.float).to(Meta.device)
        # for i in range(self.minor_semantics):
        #     score = self.base_predictors[i](user_embeddings, item_embeddings)
        #     score = score.view(batch_size, self.minor_semantics, self.minor_semantics)
        #
        #     score = score * self.masks[i].unsqueeze(0)
        #     scores += score
        # scores = scores.view(batch_size, -1)
        # scores = self.linear(scores)
        return scores.flatten()
