from model.operator.gru_operator import GRUOperator
from model.operator.transformer_operator import TransformerOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender


class PLMNRLSTURModel(BaseNegRecommender):
    news_encoder_class = TransformerOperator
    user_encoder_class = GRUOperator
