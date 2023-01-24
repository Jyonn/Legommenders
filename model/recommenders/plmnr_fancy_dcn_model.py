from model.operator.transformer_operator import TransformerOperator
from model.recommenders.fancy_dcn_model import FancyDCNModel, FancyDCNModelConfig


class PLMNRFancyDCNModel(FancyDCNModel):
    news_encoder_class = TransformerOperator

    def __init__(self, config: FancyDCNModelConfig, **kwargs):
        super().__init__(input_dim=config.hidden_size, config=config, **kwargs)
