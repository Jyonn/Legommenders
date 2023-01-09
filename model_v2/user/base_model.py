from model_v2.common.base_config import BaseUserConfig
from model_v2.common.base_encoder_model import BaseEncoderModel
from model_v2.interaction.base_interaction import BaseInteraction


class BaseUserModel(BaseEncoderModel):
    config: BaseUserConfig
    interaction: BaseInteraction
    use_neg_sampling = True

    def forward(self, clicks, **kwargs):
        raise NotImplementedError
