import copy

import torch
from torch.utils.data import Dataset as BaseDataset

from loader.data_hub import DataHub
from utils.printer import printer, Color
from utils.timer import Timer


class DataSet(BaseDataset):
    def __init__(
            self,
            hub: DataHub,
            resampler=None,
    ):
        """

        @rtype: object
        """
        self.print = printer[(self.__class__.__name__, '·', Color.GREEN)]

        self.hub = hub
        self.depot = hub.depot
        self.order = hub.order
        self.append = hub.append

        self.resampler = resampler

        self.sample_size = self.depot.sample_size
        self.split_range = (0, self.sample_size)

        self.timer = Timer(activate=True)

        # if self.resampler:
        #     data = self.depot.data[self.resampler.column_map.candidate_col]
        #     if not isinstance(data, list):
        #         data = data.tolist()
        #     self.fast_candidate_col = torch.tensor(data).unsqueeze(1)

    def __getitem__(self, index):
        index += self.split_range[0]
        return self.pack_sample(index)

    def __len__(self):
        mode_range = self.split_range
        return mode_range[1] - mode_range[0]

    def pack_sample(self, index):
        # if self.resampler and self.resampler.legommender.cacher.user.cached:
        #     self.timer.run('pack_sample')
        #     index = self.depot._indexes[index]
        #     cols = [
        #         self.resampler.column_map.label_col,
        #         self.resampler.column_map.group_col,
        #         self.resampler.column_map.user_col,
        #         # self.resampler.column_map.candidate_col,
        #     ]
        #     sample = {col: self.depot.data[col][index] for col in cols}
        #     sample[self.resampler.column_map.candidate_col] = self.fast_candidate_col[index]
        #     if self.resampler:
        #         sample = self.resampler(sample)
        #     return sample
        sample = self.depot[index]
        sample = {col: copy.copy(sample[col]) for col in [*self.order, *self.append]}
        if self.resampler:
            sample = self.resampler(sample)
        return sample

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
