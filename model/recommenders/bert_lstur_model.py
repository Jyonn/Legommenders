from model.operator.bert_operator import BertOperator
from model.operator.gru_operator import GRUOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender


class BERTLSTURModel(BaseNegRecommender):
    news_encoder_class = BertOperator
    user_encoder_class = GRUOperator
