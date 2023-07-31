from model.operator.fastformer_operator import FastformerOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender


class FastformerModel(BaseNegRecommender):
    news_encoder_class = FastformerOperator
    user_encoder_class = FastformerOperator
