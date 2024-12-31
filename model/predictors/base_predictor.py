from model.common.base_module import BaseModule
from model.lego_config import LegoConfig


class BasePredictorConfig:
    def __init__(
            self,
            hidden_size,
            embed_hidden_size,
            **kwargs
    ):
        self.hidden_size = hidden_size
        self.embed_hidden_size = embed_hidden_size


class BasePredictor(BaseModule):
    allow_ranking = True
    allow_matching = True
    keep_input_dim = False

    config_class = BasePredictorConfig

    def __init__(self, config: BasePredictorConfig, lego_config: LegoConfig):
        super().__init__()

        self.config = config
        self.lego_config: LegoConfig = lego_config

    def predict(self, user_embeddings, item_embeddings):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
