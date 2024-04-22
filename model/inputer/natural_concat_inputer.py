from collections import OrderedDict
from typing import Dict

import torch

from loader.meta import Meta
from model.inputer.base_inputer import BaseInputer
from model.inputer.concat_inputer import Pointer


class NaturalConcatInputer(BaseInputer):
    vocab = None
    special_col = 'natural_cat'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.start_prompt, self.col_prompt_map = self.get_prompt()

        self.max_content_len = self.get_max_content_len()
        self.max_sequence_len = self.max_content_len + len(self.start_prompt)
        for col in self.order:
            self.max_sequence_len += len(self.col_prompt_map[col])

    @staticmethod
    def get_start_prompt():
        raise NotImplementedError

    @staticmethod
    def get_col_prompts():
        raise NotImplementedError

    class ColPromptWrapper:
        def __init__(self, prompt_map):
            self._map = {k: torch.LongTensor(v) for k, v in prompt_map.items()}

        def __getitem__(self, col):
            brief_col = col.replace('-llama', '')
            brief_col = brief_col.replace('-token', '')
            brief_col = brief_col.replace('-bert', '')
            return self._map[brief_col]

    @classmethod
    def get_prompt(cls):
        start_prompt = cls.get_start_prompt()
        start_prompt = torch.LongTensor(start_prompt)

        # col_prompt_map = cls.get_col_prompts()
        # col_prompt_map = {k: torch.LongTensor(v) for k, v in col_prompt_map.items()}
        col_prompt_map = cls.ColPromptWrapper(cls.get_col_prompts())
        return start_prompt, col_prompt_map

    def get_max_content_len(self):
        length = 0
        for col in self.order:
            length += self.depot.cols[col].max_length or 1
        return length

    def get_empty_input(self):
        return torch.ones(self.max_sequence_len, dtype=torch.long) * Meta.UNSET

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

            pointer.update_input(input_id, self.col_prompt_map[col])
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

        seq = input_ids[self.special_col].to(Meta.device)  # type: torch.Tensor # [B, L]
        mask = (seq > Meta.UNSET).long().to(Meta.device)  # type: torch.Tensor  # [B, L]
        seq *= mask

        embedding = self.embedding_manager(self.special_col)(seq)
        embedding *= mask.unsqueeze(-1)

        return embedding
