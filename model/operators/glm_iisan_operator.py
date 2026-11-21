import abc
import os

import torch
from transformers import GlmModel

from model.common.glm_interface import ChatGLMModel
from model.operators.iisan_operator import IISANOperator


class GLMIISANOperator(IISANOperator, abc.ABC):
    dtype = torch.bfloat16

    def get_layer_nums(self):
        if isinstance(self.transformer, GlmModel):
            return super().get_layer_nums()
        return int(self.transformer.config.num_layers)
    transformer: ChatGLMModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer: GlmModel | ChatGLMModel
        self._uses_native_glm = isinstance(self.transformer, GlmModel)
        if not self._uses_native_glm:
            self.transformer = self.transformer.transformer
        self.transformer.set_input_embeddings(None)

        if self.transformer.config.hidden_size != self.config.input_dim:
            raise ValueError(f'In {self.classname}, hidden_size of transformer ({self.transformer.config.hidden_size}) '
                             f'does not match input_dim ({self.config.input_dim})')

        self._load_hidden_states()

    def _get_cache_path(self):
        if self._uses_native_glm:
            return os.path.join(self._cache_base_dir, 'states-glm.npy')
        return super()._get_cache_path()


class GLM4TH9BIISANOperator(GLMIISANOperator):
    pass
