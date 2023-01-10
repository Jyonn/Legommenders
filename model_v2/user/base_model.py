from model_v2.common.base_config import BaseConfig
from model_v2.common.base_encoder_model import BaseEncoderModel
from model_v2.interaction.base_interaction import BaseInteraction
from model_v2.utils.nr_depot import NRDepot


class NegativeSamplingConfig:
    def __init__(self, neg_count=4, neg_col='neg', **kwargs):
        self.neg_count = neg_count
        self.neg_col = neg_col


class BaseUserModel(BaseEncoderModel):
    interaction: BaseInteraction
    use_neg_sampling = True

    def __init__(self, config, nrd: NRDepot):
        super().__init__(config, nrd)

        self.negative_sampling = None
        if config.negative_sampling:
            self.negative_sampling = NegativeSamplingConfig(**config.negative_sampling)

    def forward(self, clicks, **kwargs):
        raise NotImplementedError
