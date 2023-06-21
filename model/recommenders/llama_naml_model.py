from model.operator.ada_operator import AdaOperator
from model.operator.llama_operator import LlamaOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender


class LLAMANAMLModel(BaseNegRecommender):
    news_encoder_class = LlamaOperator
    user_encoder_class = AdaOperator
