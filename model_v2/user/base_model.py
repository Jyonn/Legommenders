from oba import Obj

from model_v2.common.base_config import BaseConfig
from model_v2.common.base_encoder_model import BaseEncoderModel
from model_v2.interaction.base_interaction import BaseInteraction
from model_v2.utils.nr_depot import NRDepot


class BaseUserModel(BaseEncoderModel):
    interaction: BaseInteraction
    use_neg_sampling = True

    def __init__(self, config, nrd: NRDepot):
        super().__init__(config, nrd)

        self.neg_count = config.neg_count

    def forward(self, embeddings, mask=None, **kwargs):
        raise NotImplementedError
