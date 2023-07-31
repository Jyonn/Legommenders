from typing import Type

import torch
from torch import nn

from loader.global_setting import Setting
from model.common.user_plugin import UserPlugin
from model.operator.base_llm_operator import BaseLLMOperator
from model.operator.base_operator import BaseOperator
from model.utils.column_map import ColumnMap
from loader.embedding.embedding_manager import EmbeddingManager
from model.utils.nr_depot import NRDepot
from utils.pagers.fast_doc_pager import FastDocPager
from utils.pagers.fast_user_pager import FastUserPager
from utils.printer import printer, Color
from utils.shaper import Shaper
from utils.timer import Timer


class BaseRecommenderConfig:
    def __init__(
            self,
            hidden_size,
            user_config,
            embed_hidden_size=None,
            news_config=None,
            use_news_content: bool = True,
            max_news_content_batch_size: int = 0,
            same_dim_transform: bool = True,
            page_size: int = 512,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.news_config = news_config
        self.user_config = user_config
        self.use_news_content = use_news_content
        self.embed_hidden_size = embed_hidden_size or hidden_size

        self.max_news_content_batch_size = max_news_content_batch_size
        self.same_dim_transform = same_dim_transform

        self.page_size = page_size

        if self.use_news_content and not self.news_config:
            raise ValueError('news_config is required when use_news_content is True')


class BaseRecommender(nn.Module):
    config_class = BaseRecommenderConfig
    news_encoder_class = None  # type: Type[BaseOperator]
    user_encoder_class = None  # type: Type[BaseOperator]
    use_neg_sampling = True

    def __init__(
            self,
            config: BaseRecommenderConfig,
            column_map: ColumnMap,
            embedding_manager: EmbeddingManager,
            user_nrd: NRDepot,
            news_nrd: NRDepot,
            user_plugin: UserPlugin = None,
    ):
        super().__init__()

        self.config = config  # type: BaseRecommenderConfig
        self.print = printer[(self.__class__.__name__, '|', Color.MAGENTA)]

        self.timer = Timer(activate=True)

        self.column_map = column_map  # type: ColumnMap
        self.embedding_manager = embedding_manager
        self.embedding_table = embedding_manager.get_table()

        self.user_col = column_map.user_col
        self.clicks_col = column_map.clicks_col
        self.candidate_col = column_map.candidate_col
        self.label_col = column_map.label_col
        self.clicks_mask_col = column_map.clicks_mask_col

        self.user_config = self.user_encoder_class.config_class(**self.combine_config(
            config=self.config.user_config,
            hidden_size=config.hidden_size,
            embed_hidden_size=config.embed_hidden_size,
        ))

        self.user_encoder = self.user_encoder_class(
            config=self.user_config,
            nrd=user_nrd,
            embedding_manager=embedding_manager,
            target_user=True,
        )
        if self.config.use_news_content:
            self.news_config = self.news_encoder_class.config_class(**self.combine_config(
                config=self.config.news_config,
                hidden_size=config.hidden_size,
                embed_hidden_size=config.embed_hidden_size,
            ))

            self.news_encoder = self.news_encoder_class(
                config=self.news_config,
                nrd=news_nrd,
                embedding_manager=embedding_manager,
                target_user=False,
            )

        self.user_plugin = user_plugin
        self.shaper = Shaper()

        # fast evaluation by caching news and user representations
        self.fast_doc_eval = False
        self.fast_doc_repr = None
        self.use_fast_doc_caching = True

        self.fast_user_eval = False
        self.fast_user_repr = None
        self.use_fast_user_caching = True

        # special cases for llama
        self.llm_skip = False
        if self.config.use_news_content:
            if isinstance(self.news_encoder, BaseLLMOperator):
                if self.news_encoder.config.layer_split:
                    self.llm_skip = True
                    self.print("LLM SKIP")

    @staticmethod
    def combine_config(config: dict, **kwargs):
        for k, v in kwargs.items():
            if k not in config:
                config[k] = v
        return config

    def timing(self, activate=True):
        self.timer.activate = activate

    @staticmethod
    def get_sample_size(news_content):
        if isinstance(news_content, torch.Tensor):
            return news_content.shape[0]
        assert isinstance(news_content, dict)
        key = list(news_content.keys())[0]
        return news_content[key].shape[0]

    def get_news_content(self, batch, col):
        if self.fast_doc_eval:
            return self.fast_doc_repr[batch[col]]

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
        if self.fast_user_eval:
            return self.fast_user_repr[batch[self.user_col]]

        self.timer.run('news')
        if self.config.use_news_content:
            clicks = self.get_news_content(batch, self.clicks_col)
        else:
            clicks = self.user_encoder.inputer.get_embeddings(batch[self.clicks_col])
        self.timer.run('news')

        self.timer.run('user')
        user_embedding = self.user_encoder(
            clicks,
            mask=batch[self.clicks_mask_col].to(Setting.device),
        )
        user_embedding = self.fuse_user_plugin(batch, user_embedding)
        self.timer.run('user')
        return user_embedding

    def fuse_user_plugin(self, batch, user_embedding):
        if self.user_plugin:
            return self.user_plugin(batch[self.user_col], user_embedding)
        return user_embedding

    def forward(self, batch):
        if isinstance(batch[self.candidate_col], torch.Tensor) and batch[self.candidate_col].dim() == 1:
            batch[self.candidate_col] = batch[self.candidate_col].unsqueeze(1)

        if self.config.use_news_content:
            candidates = self.get_news_content(batch, self.candidate_col)
        else:
            candidates = self.embedding_manager(self.clicks_col)(batch[self.candidate_col].to(Setting.device))

        user_embedding = self.get_user_content(batch)

        self.timer.run('predict')
        results = self.predict(user_embedding, candidates, batch)
        self.timer.run('predict')
        return results

    def predict(self, user_embedding, candidates, batch) -> [torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def start_caching_doc_repr(self, doc_list):
        if not self.config.use_news_content:
            return
        if not Setting.fast_eval:
            return
        if not self.use_fast_doc_caching:
            return
        if self.fast_doc_eval:
            return

        self.print("Start caching doc repr")

        pager = FastDocPager(
            inputer=self.news_encoder.inputer,
            contents=doc_list,
            model=self.news_encoder,
            page_size=self.config.page_size,
            hidden_size=self.config.hidden_size,
            llm_skip=self.llm_skip,
        )

        pager.run()

        self.fast_doc_eval = True
        self.fast_doc_repr = pager.fast_doc_repr

    def end_caching_doc_repr(self):
        self.fast_doc_eval = False
        self.fast_doc_repr = None

    def start_caching_user_repr(self, user_list):
        if not Setting.fast_eval:
            return
        if not self.use_fast_user_caching:
            return
        if self.fast_user_eval:
            return

        self.print("Start caching user repr")

        pager = FastUserPager(
            contents=user_list,
            model=self.get_user_content,
            page_size=self.config.page_size,
            hidden_size=self.config.hidden_size,
        )

        pager.run()

        self.fast_user_eval = True
        self.fast_user_repr = pager.fast_user_repr

    def end_caching_user_repr(self):
        self.fast_user_eval = False
        self.fast_user_repr = None

    def parameter_split(self):
        news_names, news_parameters = self.news_encoder.get_pretrained_parameters(prefix='news_encoder')
        rec_parameters = self.named_parameters()
        rec_parameters = filter(lambda p: p[1].requires_grad, rec_parameters)
        rec_parameters = filter(lambda p: p[0] not in news_names, rec_parameters)
        rec_parameters = map(lambda p: p[1], rec_parameters)
        return news_parameters, rec_parameters

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()
