from model.operator.ada_operator import AdaOperator
from model.operator.attention_operator import AttentionOperator
from model.operator.transformer_operator import TransformerOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender


class PLMNRNAMLModel(BaseNegRecommender):
    news_encoder_class = TransformerOperator
    user_encoder_class = AdaOperator
