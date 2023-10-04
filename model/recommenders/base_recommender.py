from typing import Type

import torch
from torch import nn

from loader.global_setting import Setting
from model.common.user_plugin import UserPlugin
from model.operators.base_llm_operator import BaseLLMOperator
from model.operators.base_operator import BaseOperator
from model.predictors.base_predictor import BasePredictor
from model.utils.cacher import Cacher
from model.utils.column_map import ColumnMap
from loader.embedding.embedding_manager import EmbeddingManager
from model.utils.nr_depot import NRDepot
from utils.function import combine_config
from utils.printer import printer, Color
from utils.shaper import Shaper


class RecommenderMeta:
    def __init__(
            self,
            item_encoder_class: Type[BaseOperator],
            user_encoder_class: Type[BaseOperator],
            predictor_class: Type[BasePredictor],
    ):
        self.item_encoder_class = item_encoder_class
        self.user_encoder_class = user_encoder_class
        self.predictor_class = predictor_class


class BaseRecommenderConfig:
    def __init__(
            self,
            hidden_size,
            user_config,
            use_neg_sampling: bool = True,
            neg_count: int = 4,
            embed_hidden_size=None,
            news_config=None,
            predictor_config=None,
            use_news_content: bool = True,
            max_news_content_batch_size: int = 0,
            same_dim_transform: bool = True,
            page_size: int = 512,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.news_config = news_config
        self.user_config = user_config
        self.predictor_config = predictor_config or {}

        self.use_neg_sampling = use_neg_sampling
        self.neg_count = neg_count
        self.use_news_content = use_news_content
        self.embed_hidden_size = embed_hidden_size or hidden_size

        self.max_news_content_batch_size = max_news_content_batch_size
        self.same_dim_transform = same_dim_transform

        self.page_size = page_size

        if self.use_news_content and not self.news_config:
            raise ValueError('news_config is required when use_news_content is True')


class BaseRecommender(nn.Module):
    config_class = BaseRecommenderConfig
    # news_encoder_class = None  # type: Type[BaseOperator]
    # user_encoder_class = None  # type: Type[BaseOperator]
    # predictor_class = None  # type: Type[BasePredictor]

    # use_neg_sampling = True

    def __init__(
            self,
            meta: RecommenderMeta,
            config: BaseRecommenderConfig,
            column_map: ColumnMap,
            embedding_manager: EmbeddingManager,
            user_nrd: NRDepot,
            news_nrd: NRDepot,
            user_plugin: UserPlugin = None,
    ):
        super().__init__()

        """initializing basic attributes"""
        self.meta = meta
        self.news_encoder_class = meta.item_encoder_class
        self.user_encoder_class = meta.user_encoder_class
        self.predictor_class = meta.predictor_class

        self.use_neg_sampling = config.use_neg_sampling
        self.neg_count = config.neg_count

        self.config = config  # type: BaseRecommenderConfig
        self.print = printer[(self.__class__.__name__, '|', Color.MAGENTA)]

        self.embedding_manager = embedding_manager
        self.embedding_table = embedding_manager.get_table()

        self.user_nrd = user_nrd
        self.news_nrd = news_nrd

        self.column_map = column_map  # type: ColumnMap
        self.user_col = column_map.user_col
        self.clicks_col = column_map.clicks_col
        self.candidate_col = column_map.candidate_col
        self.label_col = column_map.label_col
        self.clicks_mask_col = column_map.clicks_mask_col

        """initializing core components"""
        self.user_encoder = self.prepare_user_module()
        self.news_encoder = None
        if self.config.use_news_content:
            self.news_encoder = self.prepare_item_module()
        self.predictor = self.prepare_predictor()

        """initializing extra components"""
        self.user_plugin = user_plugin

        """initializing utils"""
        # special cases for llama
        self.llm_skip = False
        if self.config.use_news_content:
            if isinstance(self.news_encoder, BaseLLMOperator):
                if self.news_encoder.config.layer_split:
                    self.llm_skip = True
                    self.print("LLM SKIP")

        self.shaper = Shaper()
        self.cacher = Cacher(self)

        self.loss_func = nn.CrossEntropyLoss() if self.use_neg_sampling else nn.BCEWithLogitsLoss()

    @staticmethod
    def get_sample_size(news_content):
        if isinstance(news_content, torch.Tensor):
            return news_content.shape[0]
        assert isinstance(news_content, dict)
        key = list(news_content.keys())[0]
        return news_content[key].shape[0]

    def get_news_content(self, batch, col):
        if self.cacher.fast_doc_eval:
            return self.cacher.fast_doc_repr[batch[col]]

        if not self.llm_skip:
            _shape = None
            news_content = self.shaper.transform(batch[col])  # batch_size, click_size, max_seq_len
            attention_mask = self.news_encoder.inputer.get_mask(news_content)
            news_content = self.news_encoder.inputer.get_embeddings(news_content)
        else:
            _shape = batch[col].shape
            news_content = batch[col].reshape(-1)
            attention_mask = None

        sample_size = self.get_sample_size(news_content)
        allow_batch_size = self.config.max_news_content_batch_size or sample_size
        batch_num = (sample_size + allow_batch_size - 1) // allow_batch_size

        # news_contents = torch.zeros(sample_size, self.config.hidden_size, dtype=torch.float).to(Setting.device)
        news_contents = self.news_encoder.get_full_news_placeholder(sample_size).to(Setting.device)
        for i in range(batch_num):
            start = i * allow_batch_size
            end = min((i + 1) * allow_batch_size, sample_size)
            mask = None if attention_mask is None else attention_mask[start:end]
            content = self.news_encoder(news_content[start:end], mask=mask)
            news_contents[start:end] = content

        if not self.llm_skip:
            news_contents = self.shaper.recover(news_contents)
        else:
            news_contents = news_contents.view(*_shape, -1)
        return news_contents

    def get_user_content(self, batch):
        if self.cacher.fast_user_eval:
            return self.cacher.fast_user_repr[batch[self.user_col]]

        if self.config.use_news_content:
            clicks = self.get_news_content(batch, self.clicks_col)
        else:
            clicks = self.user_encoder.inputer.get_embeddings(batch[self.clicks_col])

        user_embedding = self.user_encoder(
            clicks,
            mask=batch[self.clicks_mask_col].to(Setting.device),
        )
        user_embedding = self.fuse_user_plugin(batch, user_embedding)
        return user_embedding

    def fuse_user_plugin(self, batch, user_embedding):
        if self.user_plugin:
            return self.user_plugin(batch[self.user_col], user_embedding)
        return user_embedding

    def forward(self, batch):
        if isinstance(batch[self.candidate_col], torch.Tensor) and batch[self.candidate_col].dim() == 1:
            batch[self.candidate_col] = batch[self.candidate_col].unsqueeze(1)

        if self.config.use_news_content:
            item_embeddings = self.get_news_content(batch, self.candidate_col)
        else:
            item_embeddings = self.embedding_manager(self.clicks_col)(batch[self.candidate_col].to(Setting.device))

        user_embeddings = self.get_user_content(batch)

        # if self.use_neg_sampling:
        #     batch_size, candidate_size, _ = item_embeddings.shape
        #     user_embeddings = user_embeddings.unsqueeze(1).repeat(1, candidate_size, 1)  # B, K+1, D
        #     labels = torch.zeros(batch_size, dtype=torch.long, device=Setting.device)
        #     loss_func = nn.CrossEntropyLoss()
        # else:
        #     item_embeddings = item_embeddings.squeeze(1)
        #     labels = batch[self.label_col].float().to(Setting.device)
        #     loss_func = nn.BCEWithLogitsLoss()
        # scores = self.predictor(user_embeddings, item_embeddings)
        if self.use_neg_sampling:
            scores = self.predict_for_neg_sampling(item_embeddings, user_embeddings)
            labels = torch.zeros(scores.shape[0], dtype=torch.long, device=Setting.device)
        else:
            scores = self.predict_for_ranking(item_embeddings, user_embeddings)
            labels = batch[self.label_col].float().to(Setting.device)

        if Setting.status.is_testing or (Setting.status.is_evaluating and not Setting.simple_dev):
            return scores
        return self.loss_func(scores, labels)

    def predict_for_neg_sampling(self, item_embeddings, user_embeddings):
        batch_size, candidate_size, hidden_size = item_embeddings.shape
        if self.predictor.keep_input_dim:
            return self.predictor(user_embeddings, item_embeddings)
        user_embeddings = user_embeddings.unsqueeze(1).repeat(1, candidate_size, 1)  # B, K+1, D
        user_embeddings = user_embeddings.view(-1, hidden_size)
        item_embeddings = item_embeddings.view(-1, hidden_size)
        scores = self.predictor(user_embeddings, item_embeddings)
        scores = scores.view(batch_size, -1)
        return scores

    def predict_for_ranking(self, item_embeddings, user_embeddings):
        item_embeddings = item_embeddings.squeeze(1)
        scores = self.predictor(user_embeddings, item_embeddings)
        return scores

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()

    def prepare_user_module(self):
        user_config = self.user_encoder_class.config_class(**combine_config(
            config=self.config.user_config,
            hidden_size=self.config.hidden_size,
            embed_hidden_size=self.config.embed_hidden_size,
            input_dim=self.config.hidden_size,
        ))

        return self.user_encoder_class(
            config=user_config,
            nrd=self.user_nrd,
            embedding_manager=self.embedding_manager,
            target_user=True,
        )

    def prepare_item_module(self):
        item_config = self.news_encoder_class.config_class(**combine_config(
            config=self.config.news_config,
            hidden_size=self.config.hidden_size,
            embed_hidden_size=self.config.embed_hidden_size,
            input_dim=self.config.embed_hidden_size,
        ))

        return self.news_encoder_class(
            config=item_config,
            nrd=self.news_nrd,
            embedding_manager=self.embedding_manager,
            target_user=False,
        )

    def prepare_predictor(self):
        if self.config.use_neg_sampling and not self.predictor_class.allow_matching:
            raise ValueError(f'{self.predictor_class.__name__} does not support negative sampling')

        if not self.config.use_neg_sampling and not self.predictor_class.allow_ranking:
            raise ValueError(f'{self.predictor_class.__name__} only supports negative sampling')

        predictor_config = self.predictor_class.config_class(**combine_config(
            config=self.config.predictor_config,
            hidden_size=self.config.hidden_size,
            embed_hidden_size=self.config.embed_hidden_size,
        ))

        return self.predictor_class(config=predictor_config)

    def get_parameters(self):
        pretrained_parameters = []
        other_parameters = []
        pretrained_signals = self.news_encoder.get_pretrained_parameter_names()

        pretrained_names = []
        other_names = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_pretrained = False
            for pretrained_name in pretrained_signals:
                if name.startswith(f'news_encoder.{pretrained_name}'):
                    pretrained_names.append((name, param.data.shape))
                    pretrained_parameters.append(param)
                    is_pretrained = True
                    break

            if not is_pretrained:
                # self.print(f'[N] {name} {param.data.shape}')
                other_names.append((name, param.data.shape))
                other_parameters.append(param)

        for name, shape in pretrained_names:
            self.print(f'[P] {name} {shape}')
        for name, shape in other_names:
            self.print(f'[N] {name} {shape}')

        return pretrained_parameters, other_parameters
