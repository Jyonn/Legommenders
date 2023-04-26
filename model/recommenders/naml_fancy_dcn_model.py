from model.operator.ada_operator import AdaOperator
from model.operator.cnn_operator import CNNOperator
from model.recommenders.fancy_dcn_model import FancyDCNModel, FancyDCNModelConfig


class NAMLFancyDCNModel(FancyDCNModel):
    news_encoder_class = CNNOperator
    user_encoder_class = AdaOperator

    def __init__(self, config: FancyDCNModelConfig, **kwargs):
        super().__init__(input_dim=config.hidden_size, config=config, **kwargs)
