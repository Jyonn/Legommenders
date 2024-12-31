from collections import OrderedDict
from typing import Dict

import numpy as np
import torch

from loader.env import Env
from loader.ut.lego_ut import LegoUT
from model.inputer.base_inputer import BaseInputer


class SemanticInputer(BaseInputer):
    output_single_sequence = False

    def __init__(self, item_ut: LegoUT, item_inputs, **kwargs):
        self.user_inputs = kwargs['user_inputs']
        self.user_ut: LegoUT = kwargs['user_ut']
        assert len(self.user_inputs) == 1, 'semantic inputer only support one column of user history'
        self.history_col = self.user_inputs[0]
        self.item_ut: LegoUT = item_ut
        assert len(item_inputs) == 1, 'semantic inputer only support one column of item'
        self.semantic_col = item_inputs[0]

        super().__init__(**kwargs)

        self.max_content_len = self.get_max_content_len()  # matrix width
        self.max_sequence_len = self.get_max_sequence_len()  # matrix height

    def get_max_content_len(self):
        return self.item_ut.meta.jobs[self.semantic_col].max_len

    def get_max_sequence_len(self):
        return self.user_ut.meta.jobs[self.history_col].max_len

    def get_empty_input(self):
        return torch.zeros(self.max_sequence_len, self.max_content_len, dtype=torch.long)

    def sample_rebuilder(self, sample: OrderedDict):
        input_ids = self.get_empty_input()

        items = sample[self.history_col]
        for index, item_id in enumerate(items):
            item_sample = self.item_ut[item_id]
            value = item_sample[self.semantic_col]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            value = torch.tensor(value, dtype=torch.long)
            input_ids[index, :] = value

        attention_mask = torch.tensor([1] * len(items) + [0] * (self.max_sequence_len - len(items)), dtype=torch.long)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def get_mask(self, batched_samples: Dict[str, torch.Tensor]):
        return batched_samples['attention_mask']

    def get_embeddings(
            self,
            batched_samples: Dict[str, torch.Tensor],
    ):
        input_ids = batched_samples['input_ids'].to(Env.device)
        embedding = self.embedding_hub(self.semantic_col)(input_ids)
        return embedding
