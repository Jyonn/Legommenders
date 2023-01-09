from model_v2.common.base_config import BaseConfig
from model_v2.common.base_encoder_model import BaseEncoderModel, BaseEncoderConfig
from model_v2.interaction.base_interaction import BaseInteraction


class NegativeSamplingConfig:
    def __init__(self, neg_count=4, neg_col='neg', **kwargs):
        self.neg_count = neg_count
        self.neg_col = neg_col


class BaseUserConfig(BaseEncoderConfig):
    encoder_config_class = BaseConfig

    def __init__(
            self,
            negative_sampling=None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.negative_sampling = negative_sampling
        if negative_sampling:
            self.negative_sampling = NegativeSamplingConfig(**negative_sampling)


class BaseUserModel(BaseEncoderModel):
    config_class = BaseUserConfig
    interaction: BaseInteraction
    use_neg_sampling = True

    def forward(self, clicks, **kwargs):
        raise NotImplementedError
