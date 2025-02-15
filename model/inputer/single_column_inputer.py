from collections import OrderedDict

import numpy as np
import torch

from loader.env import Env
from model.inputer.base_inputer import BaseInputer


class SingleColumnInputer(BaseInputer):
    output_single_sequence = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert len(self.inputs) == 1, 'single column inputer only support one column in order'
        self.column = self.inputs[0]
        self.vocab = self.ut.meta.jobs[self.column].tokenizer.vocab.name

    def sample_rebuilder(self, sample: OrderedDict):
        value = sample[self.column]
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return torch.tensor(value, dtype=torch.long)

    def get_embeddings(
            self,
            batched_samples: torch.Tensor,
    ):
        embedding = self.embedding_hub(self.vocab, col_name=self.column)(batched_samples.to(Env.device))
        return embedding

    def get_mask(self, batched_samples: torch.Tensor):
        return torch.ones(batched_samples.shape, dtype=torch.long)
