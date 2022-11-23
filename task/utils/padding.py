from collections import OrderedDict

import torch
from UniTok import UniDep

from loader.global_setting import Setting


class Padding:
    def __init__(self, depot: UniDep, order: list):
        self.depot = depot
        self.order = order

    @classmethod
    def pad(cls, l: list, max_len: int):
        return l + [Setting.UNSET] * (max_len - len(l))

    def create(self, sample: OrderedDict):
        for col in sample:
            max_len = self.depot.get_max_length(col)
            if max_len:
                sample[col] = self.pad(sample[col], max_len)
            sample[col] = torch.tensor(sample[col])
        return sample

    def sample_wise_pad(self):
        sample = OrderedDict()
        for col in self.order:
            if self.depot.is_list_col(col):
                sample[col] = torch.tensor(self.pad([], self.depot.get_max_length(col)))
            else:
                sample[col] = torch.tensor(Setting.UNSET)
        return sample

    def __call__(self, sample: OrderedDict):
        return self.create(sample)
