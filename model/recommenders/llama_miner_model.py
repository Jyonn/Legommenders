from model.operator.llama_operator import LlamaOperator
from model.operator.miner_operator import PolyAttentionOperator
from model.recommenders.miner_model import MINERModel


class LLAMAMINERModel(MINERModel):
    news_encoder_class = LlamaOperator
    user_encoder_class = PolyAttentionOperator
