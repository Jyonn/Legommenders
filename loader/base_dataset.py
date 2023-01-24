import copy

from torch.utils.data import Dataset

from model.utils.nr_depot import NRDepot
from utils.printer import printer, Color


class BaseDataset(Dataset):
    def __init__(
            self,
            nrd: NRDepot,
            manager=None,
    ):
        """

        @rtype: object
        """
        self.print = printer[(self.__class__.__name__, '·', Color.GREEN)]

        self.nrd = nrd
        self.depot = nrd.depot
        self.order = nrd.order
        self.append = nrd.append
        self.append_checker()

        self.sample_size = self.depot.sample_size

        self.manager = manager  # type: Manager

        self.split_range = (0, self.sample_size)

    def append_checker(self):
        for col in self.append:
            if self.depot.is_list_col(col):
                self.print(f'{col} is a list col, please do list align in task carefully')

    def __getitem__(self, index):
        index += self.split_range[0]
        return self.pack_sample(index)

    def __len__(self):
        mode_range = self.split_range
        return mode_range[1] - mode_range[0]

    def pack_sample(self, index):
        sample = self.depot[index]
        sample = {col: copy.copy(sample[col]) for col in [*self.order, *self.append]}
        if self.manager:
            sample = self.manager.rebuild_sample(sample)
        return sample
