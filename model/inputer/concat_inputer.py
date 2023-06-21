from collections import OrderedDict
from typing import Optional, Dict, List

import torch
from UniTok import Vocab

from loader.global_setting import Setting
from model.inputer.base_inputer import BaseInputer


class Pointer:
    def __init__(self):
        self.pos = 0

    def update_input(self, input_ids, value):
        input_ids[self.pos: self.pos + len(value)] = value
        self.pos += len(value)

    def update_special_token(self, input_ids, value):
        value = torch.tensor([value], dtype=torch.long)
        return self.update_input(input_ids, value)


class ConcatInputer(BaseInputer):
    vocab = Vocab(name='__cat_inputer_special_ids')
    PAD = vocab.append('[PAD]')
    CLS = vocab.append('[CLS]')
    SEP = vocab.append('[SEP]')

    def __init__(self, use_cls_token, use_sep_token, **kwargs):
        super().__init__(**kwargs)

        self.use_cls_token = use_cls_token
        self.use_sep_token = use_sep_token

        self.max_content_len = self.get_max_content_len()
        self.max_sequence_len = self.max_content_len + int(self.use_cls_token) + int(self.use_sep_token) * len(self.order)

    def get_max_content_len(self):
        length = 0
        for col in self.order:
            length += self.depot.cols[col].max_length or 1
        return length

    def get_vocabs(self) -> Optional[List[Vocab]]:
        return [self.vocab]

    def get_empty_input(self):
        return torch.ones(self.max_sequence_len, dtype=torch.long) * Setting.UNSET

    def sample_rebuilder(self, sample: OrderedDict):
        pointer = Pointer()
        input_ids = OrderedDict()

        special_ids = self.get_empty_input()
        if self.use_cls_token:
            pointer.update_special_token(special_ids, self.CLS)

        for col in self.order:
            value = sample[col]
            if not isinstance(value, list):
                value = [value]
            value = torch.tensor(value, dtype=torch.long)

            input_id = self.get_empty_input()
            pointer.update_input(input_id, value)
            input_ids[col] = input_id

            if self.use_sep_token:
                pointer.update_special_token(special_ids, self.SEP)

        input_ids[self.vocab.name] = special_ids
        attention_mask = torch.tensor([1] * pointer.pos + [0] * (self.max_sequence_len - pointer.pos), dtype=torch.long)
        input_ids[self.vocab.name][pointer.pos:] = self.PAD

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
        shape = list(input_ids.values())[0].shape

        input_embeddings = torch.zeros(
            *shape,
            self.embedding_manager.hidden_size,
            dtype=torch.float
        ).to(Setting.device)

        for col in input_ids:
            seq = input_ids[col].to(Setting.device)  # type: torch.Tensor # [B, L]
            mask = (seq > Setting.UNSET).long().to(Setting.device)  # type: torch.Tensor  # [B, L]
            seq *= mask

            embedding = self.embedding_manager(col)(seq)
            embedding *= mask.unsqueeze(-1)

            input_embeddings += embedding
        return input_embeddings
