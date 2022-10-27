from collections import OrderedDict

import torch
from UniTok import UniDep, Vocab

from loader.global_setting import Setting


class Sequencer:
    vocab = Vocab(name='__sequencer')
    CLS = vocab.append('[CLS]')
    SEP = vocab.append('[SEP]')

    class Pointer:
        def __init__(self):
            self.pos = 0

        def update_input(self, input_id, value):
            input_id[self.pos: self.pos + len(value)] = value
            self.pos += len(value)

        def update_special_token(self, input_id, value):
            value = [value]
            return self.update_input(input_id, value)

        def run(self):
            pass

    def __init__(self, depot: UniDep, order: list, use_cls_token, use_sep_token):
        self.depot = depot
        self.order = order
        self.max_content_len = self.get_max_content_len()

        self.use_cls_token = use_cls_token
        self.use_sep_token = use_sep_token
        self.max_sequence_len = self.max_content_len + int(self.use_cls_token) + int(self.use_sep_token) * len(order)

    def get_max_content_len(self):
        length = 0
        for col in self.order:
            length += self.depot.get_max_length(col) or 1
        return length

    def get_empty_input(self):
        return torch.ones(self.max_sequence_len, dtype=torch.long) * Setting.PAD

    def create(self, sample: OrderedDict):
        pointer = self.Pointer()
        input_ids = dict()

        special_id = self.get_empty_input()
        if self.use_cls_token:
            pointer.update_special_token(special_id, self.CLS)

        for col in sample:
            value = sample[col]
            if not isinstance(value, list):
                value = [value]
            value = torch.tensor(value, dtype=torch.long)

            input_id = self.get_empty_input()
            pointer.update_input(input_id, value)
            input_ids[col] = input_ids

            if self.use_sep_token:
                pointer.update_special_token(special_id, self.SEP)

        input_ids[self.vocab.name] = special_id

        return input_ids

    def __call__(self, sample: OrderedDict):
        return self.create(sample)
