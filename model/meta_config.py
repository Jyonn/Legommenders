from typing import Type

from pigmento import pnt

from model.operators.base_operator import BaseOperator
from model.predictors.base_predictor import BasePredictor


class LegommenderMeta:
    def __init__(
            self,
            item_encoder_class: Type[BaseOperator],
            user_encoder_class: Type[BaseOperator],
            predictor_class: Type[BasePredictor],
    ):
        self.item_encoder_class = item_encoder_class
        self.user_encoder_class = user_encoder_class
        self.predictor_class = predictor_class


class LegommenderConfig:
    def __init__(
            self,
            hidden_size,
            user_config,
            use_neg_sampling: bool = True,
            neg_count: int = 4,
            embed_hidden_size=None,
            item_config=None,
            predictor_config=None,
            use_item_content: bool = True,
            max_item_content_batch_size: int = 0,
            same_dim_transform: bool = True,
            page_size: int = 512,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.item_config = item_config
        self.user_config = user_config
        self.predictor_config = predictor_config or {}

        self.use_neg_sampling = use_neg_sampling
        self.neg_count = neg_count
        self.use_item_content = use_item_content
        self.embed_hidden_size = embed_hidden_size or hidden_size

        self.max_item_content_batch_size = max_item_content_batch_size
        self.same_dim_transform = same_dim_transform

        self.page_size = page_size

        if self.use_item_content:
            if not self.item_config:
                self.item_config = {}
                # raise ValueError('item_config is required when use_item_content is True')
                pnt('automatically set item_config to an empty dict, as use_item_content is True')
