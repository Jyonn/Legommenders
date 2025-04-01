import abc

import torch
from transformers import LlamaModel

from model.operators.iisan_operator import IISANOperator


class LlamaIISANOperator(IISANOperator, abc.ABC):
    dtype = torch.bfloat16

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer: LlamaModel
        self.transformer.embed_tokens = None

        if self.transformer.config.hidden_size != self.config.input_dim:
            raise ValueError(f'In {self.classname}, hidden_size of transformer ({self.transformer.config.hidden_size}) '
                             f'does not match input_dim ({self.config.input_dim})')

        self._load_hidden_states()


class Llama1IISANOperator(LlamaIISANOperator):
    pass


class Llama2IISANOperator(LlamaIISANOperator):
    pass


class Llama3IISANOperator(LlamaIISANOperator):
    pass
