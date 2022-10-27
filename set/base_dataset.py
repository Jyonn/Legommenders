import copy
from collections import OrderedDict
from typing import Optional

from UniTok import UniDep
from torch.utils.data import Dataset

from utils.splitter import Splitter


class BaseDataset(Dataset):
    def __init__(
            self,
            depot: UniDep,
            order: list,
            append: Optional[list] = None,
            splitter: Splitter = None,
            mode=None,
    ):
        """

        @rtype: object
        """
        self.depot = depot
        self.order = order
        self.append = self.get_append(append)

        self.mode = mode
        self.sample_size = self.depot.sample_size

        self.task = None

        if splitter is None:
            self.split_range = (0, self.sample_size)
        else:
            self.split_range = splitter.divide(self.sample_size)[self.mode]
            assert splitter.contains(self.mode)

    def get_append(self, append: list):
        append = append or []
        for col in append:
            if self.depot.is_list_col(col):
                raise ValueError(f'list column {col} cannot be appended')
        return append

    def __getitem__(self, index):
        index += self.split_range[0]
        return self.pack_sample(index)

    def __len__(self):
        mode_range = self.split_range
        return mode_range[1] - mode_range[0]

    def register_task(self, task):
        self.task = task

    def pack_sample(self, index):
        sample = self.depot[index]
        inputs = OrderedDict()
        for col in self.order:
            inputs[col] = copy.copy(sample[col])
        append = OrderedDict()
        for col in self.append:
            append[col] = sample[col]
        sample = dict(append=append, inputs=inputs)
        if self.task:
            sample = self.task.rebuild_sample(sample, self)
        return sample

    @classmethod
    def parse(cls, data, depots, splitter):
        sets = dict()
        for mode in data.split:
            sets[mode] = cls(
                order=data.order,
                append=data.append,
                depot=depots[mode],
                splitter=splitter,
                mode=mode,
            )
        return sets
