from collections import OrderedDict
from typing import Optional, List, Dict

import numpy as np
import torch
from UniTok import Vocab
from pigmento import pnt

from loader.data_hub import DataHub
from loader.meta import Meta
from model.inputer.base_inputer import BaseInputer
from model.inputer.concat_inputer import ConcatInputer, Pointer


class SemanticInputer(BaseInputer):
    output_single_sequence = False

    def __init__(self, item_hub: DataHub, **kwargs):
        self.order = kwargs['hub'].order
        self.depot = kwargs['hub'].depot
        assert len(self.order) == 1, 'semantic inputer only support one column of user history'
        self.history_col = self.order[0]
        self.item_hub = item_hub
        assert len(self.item_hub.order) == 1, 'semantic inputer only support one column of item'
        self.semantic_col = self.item_hub.order[0]

        super().__init__(**kwargs)

        self.max_content_len = self.get_max_content_len()  # matrix width
        self.max_sequence_len = self.get_max_sequence_len()  # matrix height

    def get_max_content_len(self):
        return self.item_hub.depot.cols[self.semantic_col].max_length

    def get_max_sequence_len(self):
        return self.depot.cols[self.history_col].max_length

    def get_empty_input(self):
        return torch.zeros(self.max_sequence_len, self.max_content_len, dtype=torch.long)

    def sample_rebuilder(self, sample: OrderedDict):
        input_ids = self.get_empty_input()

        items = sample[self.history_col]
        for index, item_id in enumerate(items):
            item_sample = self.item_hub.depot[item_id]
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
        input_ids = batched_samples['input_ids'].to(Meta.device)
        embedding = self.embedding_manager(self.semantic_col)(input_ids)
        return embedding
