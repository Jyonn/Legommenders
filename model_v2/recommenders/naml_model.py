from model_v2.operator.ada_operator import AdaOperator
from model_v2.operator.cnn_operator import CNNOperator
from model_v2.recommenders.base_neg_recommender import BaseNegRecommender


class NAMLModel(BaseNegRecommender):
    news_encoder_class = CNNOperator
    user_encoder_class = AdaOperator

