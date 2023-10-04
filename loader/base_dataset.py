import copy

import torch
from torch.utils.data import Dataset

from model.utils.nr_depot import NRDepot
from utils.printer import printer, Color
from utils.timer import Timer


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

        self.timer = Timer(activate=True)

        if self.manager:
            data = self.depot.data[self.manager.column_map.candidate_col]
            if not isinstance(data, list):
                data = data.tolist()
            self.fast_candidate_col = torch.tensor(data).unsqueeze(1)

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
        if self.manager and self.manager.recommender.cacher.fast_user_eval:
            self.timer.run('pack_sample')
            index = self.depot._indexes[index]
            cols = [
                self.manager.column_map.label_col,
                self.manager.column_map.group_col,
                self.manager.column_map.user_col,
                # self.manager.column_map.candidate_col,
            ]
            sample = {col: self.depot.data[col][index] for col in cols}
            sample[self.manager.column_map.candidate_col] = self.fast_candidate_col[index]
            self.timer.run('pack_sample')
            return sample
        sample = self.depot[index]
        sample = {col: copy.copy(sample[col]) for col in [*self.order, *self.append]}
        if self.manager:
            sample = self.manager.rebuild_sample(sample)
        return sample

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
