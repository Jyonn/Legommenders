import abc

import torch
from transformers import OPTModel

from model.operators.iisan_operator import IISANOperator


class OPTIISANOperator(IISANOperator, abc.ABC):
    dtype = torch.bfloat16
    transformer: OPTModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer.decoder.embed_tokens = None

        self._load_hidden_states()

    @property
    def cache_hidden_size(self):
        return self.transformer.config.hidden_size


class OPTBaseIISANOperator(OPTIISANOperator):
    pass


class OPTLargeIISANOperator(OPTIISANOperator):
    pass
