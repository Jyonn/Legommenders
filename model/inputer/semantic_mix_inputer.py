from collections import OrderedDict

import numpy as np
import torch

from loader.ut.lego_ut import LegoUT
from model.inputer.base_inputer import BaseInputer


class SemanticMixInputer(BaseInputer):
    output_single_sequence = True

    def __init__(self, item_ut: LegoUT, item_inputs, **kwargs):
        self.user_inputs = kwargs['user_inputs']
        self.user_ut = kwargs['user_ut']
        assert len(self.user_inputs) == 1, 'semantic inputer only support one column of user semantics'
        self.user_semantic_col = self.user_inputs[0]
        assert len(self.inputs) == 1, 'semantic inputer only support one column of item'
        self.semantic_col = self.inputs[0]

        super().__init__(**kwargs)

    def sample_rebuilder(self, sample: OrderedDict):
        user_semantics = sample[self.user_semantic_col]
        if isinstance(user_semantics, np.ndarray):
            user_semantics = user_semantics.tolist()
        return torch.tensor(user_semantics, dtype=torch.long)

    def get_embeddings(
            self,
            batched_samples: torch.Tensor,
    ):
        embedding = self.embedding_hub(self.semantic_col)(batched_samples)
        return embedding

    def get_mask(self, batched_samples: torch.Tensor):
        return torch.ones(batched_samples.shape, dtype=torch.long)
