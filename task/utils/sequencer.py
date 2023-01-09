from collections import OrderedDict

import torch
from UniTok import UniDep, Vocab

from loader.global_setting import Setting


class Sequencer:
    vocab = Vocab(name='__sequencer')
    PAD = vocab.append('[PAD]')
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
        return torch.ones(self.max_sequence_len, dtype=torch.long) * Setting.UNSET

    def create(self, sample: OrderedDict):
        pointer = self.Pointer()
        input_ids = OrderedDict()

        special_ids = self.get_empty_input()
        if self.use_cls_token:
            pointer.update_special_token(special_ids, self.CLS)

        for col in sample:
            value = sample[col]
            if not isinstance(value, list):
                value = [value]
            value = torch.tensor(value, dtype=torch.long)

            input_id = self.get_empty_input()
            pointer.update_input(input_id, value)
            input_ids[col] = input_id

            if self.use_sep_token:
                pointer.update_special_token(special_ids, self.SEP)

        # input_ids[self.vocab.name] = special_id
        # attention_mask = self._get_attention_mask(input_ids)
        # special_id *= attention_mask
        # return input_ids, attention_mask
        input_ids[self.vocab.name] = special_ids
        attention_mask = torch.tensor([1] * pointer.pos + [0] * (self.max_sequence_len - pointer.pos), dtype=torch.long)
        input_ids[self.vocab.name][pointer.pos:] = self.PAD
        return input_ids, attention_mask

    # @staticmethod
    # def _get_attention_mask(inputs) -> torch.Tensor:
    #     mask = None
    #     for col in inputs:
    #         seq = inputs[col]  # type: torch.Tensor
    #         if mask is None:
    #             mask = torch.zeros(*seq.shape, dtype=torch.long)
    #         col_mask = (seq > Setting.UNSET).long()
    #         mask |= col_mask
    #     return mask

    def __call__(self, sample: OrderedDict):
        return self.create(sample)
