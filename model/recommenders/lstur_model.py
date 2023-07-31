from model.operator.cnn_cat_operator import CNNCatOperator
from model.operator.gru_operator import GRUOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender


class LSTURModel(BaseNegRecommender):
    news_encoder_class = CNNCatOperator
    user_encoder_class = GRUOperator
    news_encoder: CNNCatOperator
    user_encoder: GRUOperator

    user_encoder: GRUOperator

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.user_plugin:
            user_embed_size = self.user_encoder.config.num_columns * self.user_encoder.config.hidden_size
            self.user_plugin.init_projection(user_embed_size)

        self.news_encoder.num_columns = self.user_encoder.config.num_columns
