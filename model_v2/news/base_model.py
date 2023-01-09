from model_v2.common.base_encoder_model import BaseEncoderModel, BaseEncoderConfig


class BaseNewsConfig(BaseEncoderConfig):
    pass


class BaseNewsModel(BaseEncoderModel):
    config_class = BaseNewsConfig
