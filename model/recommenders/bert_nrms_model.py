from model.operator.attention_operator import AttentionOperator
from model.operator.bert_operator import BertOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender


class BERTNRMSModel(BaseNegRecommender):
    news_encoder_class = BertOperator
    user_encoder_class = AttentionOperator
