from collections import OrderedDict
from typing import Dict

import torch

from loader.global_setting import Setting
from model.inputer.base_inputer import BaseInputer
from model.inputer.concat_inputer import Pointer


class NaturalConcatInputer(BaseInputer):
    vocab = None
    special_col = 'natural_cat'

    start_prompt = [10130, 4274, 29901]
    start_prompt = torch.LongTensor(start_prompt)

    col_map = dict(
        title=[529, 3257, 29958],
        abs=[529, 16595, 29958],
        cat=[529, 7320, 29958],
        subCat=[529, 1491, 7320, 29958],
    )
    # convert col_map to LongTensor
    col_map = {k: torch.LongTensor(v) for k, v in col_map.items()}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_content_len = self.get_max_content_len()
        self.max_sequence_len = self.max_content_len + len(self.start_prompt)
        for col in self.order:
            self.max_sequence_len += len(self.col_map[col])
        print(f'max_sequence_len: {self.max_sequence_len}')

    def get_max_content_len(self):
        length = 0
        for col in self.order:
            length += self.depot.cols[col].max_length or 1
        return length

    def get_empty_input(self):
        return torch.ones(self.max_sequence_len, dtype=torch.long) * Setting.UNSET

    def sample_rebuilder(self, sample: OrderedDict):
        pointer = Pointer()
        input_ids = OrderedDict()
        input_id = self.get_empty_input()

        pointer.update_input(input_id, self.start_prompt)

        for col in self.order:
            value = sample[col]
            if not isinstance(value, list):
                value = [value]
            value = torch.tensor(value, dtype=torch.long)

            pointer.update_input(input_id, self.col_map[col])
            pointer.update_input(input_id, value)

        input_ids[self.special_col] = input_id
        attention_mask = torch.tensor([1] * pointer.pos + [0] * (self.max_sequence_len - pointer.pos), dtype=torch.long)
        input_ids[self.special_col][pointer.pos:] = 0

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
        input_ids = batched_samples['input_ids']

        seq = input_ids[self.special_col].to(Setting.device)  # type: torch.Tensor # [B, L]
        mask = (seq > Setting.UNSET).long().to(Setting.device)  # type: torch.Tensor  # [B, L]
        seq *= mask

        embedding = self.embedding_manager(self.special_col)(seq)
        embedding *= mask.unsqueeze(-1)

        return embedding
