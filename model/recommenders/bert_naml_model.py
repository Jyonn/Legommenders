from model.operator.ada_operator import AdaOperator
from model.operator.bert_operator import BertOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender


class BERTNAMLModel(BaseNegRecommender):
    news_encoder_class = BertOperator
    user_encoder_class = AdaOperator
