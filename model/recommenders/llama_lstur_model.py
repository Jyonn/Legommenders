from model.operator.gru_operator import GRUOperator
from model.operator.llama_operator import LlamaOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender


class LLAMALSTURModel(BaseNegRecommender):
    news_encoder_class = LlamaOperator
    user_encoder_class = GRUOperator
