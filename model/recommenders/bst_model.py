from model.operator.pooling_operator import PoolingOperator
from model.operator.transformer_operator import TransformerOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender


class BSTModel(BaseNegRecommender):
    news_encoder_class = PoolingOperator
    user_encoder_class = TransformerOperator
