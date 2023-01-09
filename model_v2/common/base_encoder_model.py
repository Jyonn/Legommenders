from abc import ABC
from typing import Type

from model_v2.common.base_model import BaseModel
from model_v2.inputer.base_inputer import BaseInputer


class BaseEncoderModel(BaseModel, ABC):
    encoder_class = None  # type: Type[BaseModel]
    inputer_class = None  # type: Type[BaseInputer]

    def __init__(self, config):
        super().__init__(config)
        self.encoder = self.encoder_class(config)
        self.inputer = self.inputer_class(**config.inputer_config)

    def forward(self, embeddings, **kwargs):
        return self.encoder(embeddings)
