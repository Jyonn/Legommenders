from model_v2.news.attention_model import NewsAttentionFusionModel
from model_v2.recommenders.base_model import BaseRecommender, BaseRecommenderConfig
from model_v2.user.attention_model import UserAttentionFusionModel


class NRMSConfig(BaseRecommenderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class NRMSModel(BaseRecommender):
    config_class = NRMSConfig
    news_encoder_class = NewsAttentionFusionModel
    user_encoder_class = UserAttentionFusionModel

    def __init__(
            self,
            config: NRMSConfig,
    ):
        super().__init__(config)
