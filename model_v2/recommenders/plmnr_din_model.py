from model_v2.operator.transformer_operator import TransformerOperator
from model_v2.recommenders.dcn_model import DCNModel, DCNModelConfig
from model_v2.recommenders.din_model import DINModel


class PLMNRDINModel(DINModel):
    news_encoder_class = TransformerOperator

    def __init__(self, config: DCNModelConfig, **kwargs):
        super().__init__(input_dim=config.hidden_size, config=config, **kwargs)
