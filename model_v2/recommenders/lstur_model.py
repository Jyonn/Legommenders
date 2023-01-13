from model_v2.operator.cnn_cat_operator import CNNCatOperator
from model_v2.operator.gru_operator import GRUOperator
from model_v2.recommenders.base_neg_recommender import BaseNegRecommender


class LSTURModel(BaseNegRecommender):
    news_encoder_class = CNNCatOperator
    user_encoder_class = GRUOperator

