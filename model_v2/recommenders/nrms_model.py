from model_v2.operator.attention_operator import AttentionOperator
from model_v2.recommenders.base_neg_recommender import BaseNegRecommender


class NRMSModel(BaseNegRecommender):
    news_encoder_class = AttentionOperator
    user_encoder_class = AttentionOperator

