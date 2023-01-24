from typing import Type

from torch import nn

from loader.global_setting import Setting
from model.operator.base_operator import BaseOperator
from model.utils.column_map import ColumnMap
from loader.embedding.embedding_manager import EmbeddingManager
from model.utils.nr_depot import NRDepot
from utils.printer import printer, Color
from utils.shaper import Shaper
from utils.timer import Timer


class BaseRecommenderConfig:
    def __init__(
            self,
            hidden_size,
            user_config,
            news_config=None,
            use_news_content: bool = True,
    ):
        self.hidden_size = hidden_size
        self.news_config = news_config
        self.user_config = user_config
        self.use_news_content = use_news_content

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
    ):
        super().__init__()

        self.config = config  # type: BaseRecommenderConfig
        self.print = printer[(self.__class__.__name__, '|', Color.MAGENTA)]

        self.timer = Timer(activate=False)

        self.column_map = column_map  # type: ColumnMap
        self.embedding_manager = embedding_manager
        self.embedding_table = embedding_manager.get_table()

        self.clicks_col = column_map.clicks_col
        self.candidate_col = column_map.candidate_col
        self.label_col = column_map.label_col
        self.clicks_mask_col = column_map.clicks_mask_col

        self.user_config = self.user_encoder_class.config_class(**config.user_config)
        self.user_encoder = self.user_encoder_class(
            config=self.user_config,
            nrd=user_nrd,
            embedding_manager=embedding_manager
        )
        if self.config.use_news_content:
            self.news_config = self.news_encoder_class.config_class(**config.news_config)
            self.news_encoder = self.news_encoder_class(
                config=self.news_config,
                nrd=news_nrd,
                embedding_manager=embedding_manager,
            )

        self.shaper = Shaper()

    def timing(self, activate=True):
        self.timer.activate = activate

    def get_content(self, batch, col):
        news_content = self.shaper.transform(batch[col])
        attention_mask = self.news_encoder.inputer.get_mask(news_content)

        # batch_size * click_size, max_seq_len, embedding_dim
        news_content = self.news_encoder.inputer.get_embeddings(news_content)

        news_content = self.news_encoder(news_content, mask=attention_mask)  # batch_size * click_size, embedding_dim
        news_content = self.shaper.recover(news_content)
        return news_content

    def forward(self, batch):
        self.timer('news encoder')
        if self.config.use_news_content:
            candidates = self.get_content(batch, self.candidate_col)  # batch_size, candidate_size, embedding_dim
            clicks = self.get_content(batch, self.clicks_col)  # batch_size, click_size, embedding_dim
        else:
            candidates = self.embedding_manager(self.clicks_col)(batch[self.candidate_col].to(Setting.device))
            clicks = self.user_encoder.inputer.get_embeddings(batch[self.clicks_col])
        self.timer('news encoder')

        self.timer('user encoder')
        user_embedding = self.user_encoder(
            clicks,
            mask=batch[self.clicks_mask_col].to(Setting.device),
        )
        self.timer('user encoder')

        self.timer('interaction')
        results = self.predict(user_embedding, candidates, labels=batch[self.label_col].to(Setting.device))
        self.timer('interaction')
        return results

    def predict(self, user_embedding, candidates, labels):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()
