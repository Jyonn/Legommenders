from typing import Type, Optional

from loader.column_map import ColumnMap
from loader.embedding_hub import EmbeddingHub
from loader.ut.lego_ut import LegoUT
from model.operators.base_operator import BaseOperator
from model.predictors.base_predictor import BasePredictor
from utils.function import combine_config


class LegoConfig:
    column_map: ColumnMap
    embedding_hub: EmbeddingHub

    item_ut: LegoUT
    item_inputs: list

    user_ut: LegoUT
    user_inputs: list

    item_operator_class: Type[BaseOperator]
    user_operator_class: Type[BaseOperator]
    predictor_class: Type[BasePredictor]

    item_operator: Optional[BaseOperator]
    user_operator: BaseOperator
    predictor: BasePredictor

    def __init__(
            self,
            hidden_size,
            user_config,
            neg_count: int = 4,
            item_hidden_size=None,
            item_config=None,
            predictor_config=None,
            use_neg_sampling: bool = True,
            use_item_content: bool = True,
            use_fast_eval: bool = True,
            item_page_size: int = 0,
            cache_page_size: int = 512,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.item_config = item_config
        self.user_config = user_config
        self.predictor_config = predictor_config or {}

        self.use_item_content = use_item_content
        self.item_hidden_size = item_hidden_size or hidden_size
        if self.use_item_content:
            self.item_config = self.item_config or {}

        self.use_neg_sampling = use_neg_sampling
        self.neg_count = neg_count

        self.item_page_size = item_page_size
        self.cache_page_size = cache_page_size
        self.use_fast_eval = use_fast_eval

    def set_component_classes(
        self,
        item_operator_class: Type[BaseOperator],
        user_operator_class: Type[BaseOperator],
        predictor_class: Type[BasePredictor]
    ):
        self.item_operator_class = item_operator_class
        self.user_operator_class = user_operator_class
        self.predictor_class = predictor_class

    def set_item_ut(self, item_ut: LegoUT, item_inputs: list):
        self.item_ut = item_ut
        self.item_inputs = item_inputs

    def set_user_ut(self, user_ut: LegoUT, user_inputs: list):
        self.user_ut = user_ut
        self.user_inputs = user_inputs

    def set_column_map(self, column_map: ColumnMap):
        self.column_map = column_map

    def set_embedding_hub(self, embedding_hub: EmbeddingHub):
        self.embedding_hub = embedding_hub

    def build_components(self):
        self.item_operator = None

        if self.use_item_content:
            item_config = self.item_operator_class.config_class(**combine_config(
                config=self.item_config,
                hidden_size=self.hidden_size,
                input_dim=self.item_hidden_size,
            ))

            self.item_operator = self.item_operator_class(
                config=item_config,
                target_user=False,
                lego_config=self,
            )

        user_config = self.user_operator_class.config_class(**combine_config(
            config=self.user_config,
            hidden_size=self.hidden_size,
            input_dim=self.item_operator.output_dim if self.use_item_content else self.item_hidden_size,
        ))

        if self.user_operator_class.flatten_mode:
            user_config.inputer_config['item_ut'] = self.item_ut
            user_config.inputer_config['item_inputs'] = self.item_inputs

        self.user_operator = self.user_operator_class(
            config=user_config,
            target_user=True,
            lego_config=self,
        )

        if self.use_neg_sampling and not self.predictor_class.allow_matching:
            raise ValueError(f'{self.predictor_class.__name__} does not support negative sampling')

        if not self.use_neg_sampling and not self.predictor_class.allow_ranking:
            raise ValueError(f'{self.predictor_class.__name__} only supports negative sampling')

        predictor_config = self.predictor_class.config_class(**combine_config(
            config=self.predictor_config,
            hidden_size=self.hidden_size,
        ))

        self.predictor = self.predictor_class(
            config=predictor_config,
            lego_config=self,
        )

    def register_inputer_vocabs(self):
        if self.use_item_content:
            item_vocabs = self.item_operator.inputer.get_vocabs()
            for vocab in item_vocabs:
                self.embedding_hub.register_vocab(vocab)

        user_vocabs = self.user_operator.inputer.get_vocabs()
        for vocab in user_vocabs:
            self.embedding_hub.register_vocab(vocab)
