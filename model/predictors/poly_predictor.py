from typing import Type

import torch

from model.predictors.base_predictor import BasePredictor, BasePredictorConfig
from utils.function import combine_config


class PolyPredictorConfig(BasePredictorConfig):
    def __init__(
            self,
            num_layers,
            base_predictor: str = 'dot',
            base_predictor_config: dict = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.num_layers = num_layers
        base_predictor_config = base_predictor_config or {}

        from loader.class_hub import ClassHub
        predictors = ClassHub.predictors()
        base_predictor_class = predictors(base_predictor)  # type: Type[BasePredictor]

        self.base_predictor_config = base_predictor_class.config_class(
            **combine_config(config=base_predictor_config, **kwargs)
        )
        self.base_predictor = base_predictor_class(self.base_predictor_config)  # type: BasePredictor


class PolyPredictor(BasePredictor):
    config_class = PolyPredictorConfig
    config: PolyPredictorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.base_predictor = self.config.base_predictor

        self.linear = torch.nn.Linear(
            in_features=self.config.num_layers,
            out_features=1,
            bias=True
        )

    def predict(self, user_embeddings, item_embeddings):
        # user_embeddings: [S, B, D]
        # item_embeddings: [B, D]

        scores = []

        for i in range(user_embeddings.shape[0]):
            scores.append(self.base_predictor(user_embeddings[i], item_embeddings))  # [B]

        scores = torch.stack(scores, dim=-1)  # [B, S]
        # mean pooling
        scores = torch.mean(scores, dim=-1)  # [B]
        return scores
        # scores = self.linear(scores)  # [B, 1]
        # return scores.squeeze(-1)


