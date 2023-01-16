from model_v2.operator.pooling_operator import PoolingOperator
from model_v2.operator.transformer_operator import TransformerOperator
from model_v2.recommenders.base_neg_recommender import BaseNegRecommender


class BSTModel(BaseNegRecommender):
    news_encoder_class = PoolingOperator
    user_encoder_class = TransformerOperator
