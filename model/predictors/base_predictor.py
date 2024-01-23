from torch import nn

from model.common.base_module import BaseModule


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

    def __init__(self, config: BasePredictorConfig):
        super().__init__()

        self.config = config

    def predict(self, user_embeddings, item_embeddings):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
