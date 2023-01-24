from model.operator.attention_operator import AttentionOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender


class NRMSModel(BaseNegRecommender):
    news_encoder_class = AttentionOperator
    user_encoder_class = AttentionOperator

