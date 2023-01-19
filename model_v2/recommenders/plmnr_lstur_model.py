from model_v2.operator.attention_operator import AttentionOperator
from model_v2.operator.gru_operator import GRUOperator
from model_v2.operator.transformer_operator import TransformerOperator
from model_v2.recommenders.base_neg_recommender import BaseNegRecommender


class PLMNRLSTURModel(BaseNegRecommender):
    news_encoder_class = TransformerOperator
    user_encoder_class = GRUOperator
