from model.operator.bert_operator import BertOperator
from model.operator.fastformer_operator import FastformerOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender


class BERTFastformerModel(BaseNegRecommender):
    news_encoder_class = BertOperator
    user_encoder_class = FastformerOperator
