from model.operator.fastformer_operator import FastformerOperator
from model.operator.llama_operator import LlamaOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender


class LLAMAFastformerModel(BaseNegRecommender):
    news_encoder_class = LlamaOperator
    user_encoder_class = FastformerOperator
