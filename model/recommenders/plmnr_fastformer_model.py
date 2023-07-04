from model.operator.attention_operator import AttentionOperator
from model.operator.fastformer_operator import FastformerOperator
from model.operator.transformer_operator import TransformerOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender


class PLMNRFastformerModel(BaseNegRecommender):
    news_encoder_class = TransformerOperator
    user_encoder_class = FastformerOperator
