from model_v2.common.attention_fusion_model import AttentionFusionOperator
from model_v2.common.base_encoder_model import BaseEncoderModel
from model_v2.inputer.cat_inputer import CatInputer


class NewsAttentionFusionModel(BaseEncoderModel):
    operator_class = AttentionFusionOperator
    inputer_class = CatInputer
