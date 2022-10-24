from collections import OrderedDict

from UniTok import UniDep
from torch.utils.data import Dataset

from utils.splitter import Splitter


class BaseDataset(Dataset):
    def __init__(
            self,
            depot: UniDep,
            order: list,
            splitter: Splitter = None,
            mode=None,
    ):
        self.depot = depot
        self.order = order
        self.mode = mode
        self.sample_size = self.depot.sample_size

        self.task = None

        if splitter is None:
            self.split_range = (0, self.sample_size)
        else:
            self.split_range = splitter.divide(self.sample_size)[self.mode]
            assert splitter.contains(self.mode)

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
        d = OrderedDict()
        for col in self.order:
            d[col] = sample[col]

        if self.task.sample_static_rebuilder:
            d = self.task.sample_static_rebuilder(d)
        return d

    @classmethod
    def parse(cls, data, depots, splitter):
        sets = dict()
        for mode in data.split:
            sets[mode] = cls(
                order=data.order,
                depot=depots[mode],
                splitter=splitter,
                mode=mode,
            )
        return sets
