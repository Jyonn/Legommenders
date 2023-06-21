from model.operator.attention_operator import AttentionOperator
from model.operator.llama_operator import LlamaOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender


class LLAMANRMSModel(BaseNegRecommender):
    news_encoder_class = LlamaOperator
    user_encoder_class = AttentionOperator
