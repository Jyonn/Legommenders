from collections import OrderedDict

import numpy as np
import torch

from loader.ut.lego_ut import LegoUT
from model.inputer.base_inputer import BaseInputer


class SemanticMixInputer(BaseInputer):
    output_single_sequence = True

    def __init__(self, **kwargs):
        self.item_ut: LegoUT = kwargs['item_ut']
        self.item_inputs: list = kwargs['item_inputs']
        assert len(self.inputs) == 1, 'semantic inputer only support one column of user semantics'
        self.user_semantic_col = self.inputs[0]
        assert len(self.item_inputs) == 1, 'semantic inputer only support one column of item'
        self.semantic_col = self.item_inputs[0]

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
        vocab = self.item_ut.meta.features[self.semantic_col].tokenizer.vocab
        embedding = self.eh(vocab)(batched_samples)
        return embedding

    def get_mask(self, batched_samples: torch.Tensor):
        return torch.ones(batched_samples.shape, dtype=torch.long)
