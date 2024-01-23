from collections import OrderedDict
from typing import Optional, List

import numpy as np
import torch
from UniTok import Vocab

from loader.data_hub import DataHub
from loader.meta import Meta
from model.inputer.concat_inputer import ConcatInputer, Pointer


class FlattenSeqInputer(ConcatInputer):
    output_single_sequence = True

    vocab = Vocab(name='__flatten_seq_special_ids')
    PAD = vocab.append('[PAD]')
    CLS = vocab.append('[CLS]')
    SEP = vocab.append('[SEP]')
    ATTR_SEP = vocab.append('[ATTR_SEP]')

    def __init__(self, item_hub: DataHub, use_attr_sep_token=True, **kwargs):
        self.order = kwargs['hub'].order
        self.depot = kwargs['hub'].depot
        assert len(self.order) == 1, 'flatten seq inputer only support one column of user history'
        self.history_col = self.order[0]
        self.item_hub = item_hub
        self.use_attr_sep_token = use_attr_sep_token

        self.max_history_len = self.depot.cols[self.history_col].max_length
        super().__init__(**kwargs)

    def get_max_content_len(self):
        item_length = 0
        for col in self.item_hub.order:
            item_length += self.item_hub.depot.cols[col].max_length or 1
        return self.max_history_len * item_length

    def get_max_sequence_len(self):
        return (self.max_content_len +
                int(self.use_cls_token) +
                int(self.use_sep_token) * self.max_history_len +
                int(self.use_attr_sep_token) * self.max_history_len * (len(self.item_hub.order) - 1))

    def get_vocabs(self) -> Optional[List[Vocab]]:
        return [self.vocab]

    def get_empty_input(self):
        return torch.ones(self.max_sequence_len, dtype=torch.long) * Meta.UNSET

    def sample_rebuilder(self, sample: OrderedDict):
        pointer = Pointer()
        input_ids = OrderedDict()
        for col in self.item_hub.order:
            input_ids[col] = self.get_empty_input()

        special_ids = self.get_empty_input()
        if self.use_cls_token:
            pointer.update_special_token(special_ids, self.CLS)

        items = sample[self.history_col]
        for item_id in items:
            item_sample = self.item_hub.depot[item_id]
            for attr_index, col in enumerate(self.item_hub.order):
                value = item_sample[col]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                if not isinstance(value, list):
                    value = [value]
                value = torch.tensor(value, dtype=torch.long)

                pointer.update_input(input_ids[col], value)
                if self.use_attr_sep_token and attr_index < len(self.item_hub.order) - 1:
                    pointer.update_special_token(special_ids, self.ATTR_SEP)
            if self.use_sep_token:
                pointer.update_special_token(special_ids, self.SEP)

        input_ids[self.vocab.name] = special_ids
        attention_mask = torch.tensor([1] * pointer.pos + [0] * (self.max_sequence_len - pointer.pos), dtype=torch.long)
        input_ids[self.vocab.name][pointer.pos:] = self.PAD

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
