from model.operator.ada_operator import AdaOperator
from model.operator.cnn_operator import CNNOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender


class NAMLModel(BaseNegRecommender):
    news_encoder_class = CNNOperator
    user_encoder_class = AdaOperator

