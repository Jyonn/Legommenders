from model_v2.operator.transformer_operator import TransformerOperator
from model_v2.recommenders.base_neg_recommender import BaseNegRecommender


class PRECModel(BaseNegRecommender):
    news_encoder_class = TransformerOperator
    user_encoder_class = TransformerOperator
