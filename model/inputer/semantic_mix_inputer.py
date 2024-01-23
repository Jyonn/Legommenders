from collections import OrderedDict

import numpy as np
import torch

from loader.data_hub import DataHub
from model.inputer.base_inputer import BaseInputer


class SemanticMixInputer(BaseInputer):
    output_single_sequence = True

    def __init__(self, item_hub: DataHub, **kwargs):
        self.order = kwargs['hub'].order
        self.depot = kwargs['hub'].depot
        assert len(self.order) == 1, 'semantic inputer only support one column of user semantics'
        self.user_semantic_col = self.order[0]
        self.item_hub = item_hub
        assert len(self.item_hub.order) == 1, 'semantic inputer only support one column of item'
        self.semantic_col = self.item_hub.order[0]

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
        embedding = self.embedding_manager(self.semantic_col)(batched_samples)
        return embedding

    def get_mask(self, batched_samples: torch.Tensor):
        return torch.ones(batched_samples.shape, dtype=torch.long)
