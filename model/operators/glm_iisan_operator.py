import abc

import torch

from model.common.glm_interface import ChatGLMModel
from model.operators.iisan_operator import IISANOperator


class GLMIISANOperator(IISANOperator, abc.ABC):
    dtype = torch.bfloat16
    transformer: ChatGLMModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer: ChatGLMModel = self.transformer.transformer
        self.transformer.set_input_embeddings(None)

        if self.transformer.config.hidden_size != self.config.input_dim:
            raise ValueError(f'In {self.classname}, hidden_size of transformer ({self.transformer.config.hidden_size}) '
                             f'does not match input_dim ({self.config.input_dim})')

        self._load_hidden_states()


class GLM4TH9BIISANOperator(GLMIISANOperator):
    pass
