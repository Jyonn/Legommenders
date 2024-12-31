import copy

from torch.utils.data import Dataset as BaseDataset

from loader.ut.lego_ut import LegoUT


class DataSet(BaseDataset):
    def __init__(
            self,
            ut: LegoUT,
            resampler=None,
    ):
        self.ut = ut
        self.resampler = resampler

    def __getitem__(self, index):
        _sample = self.ut[index]
        sample = dict()
        for col in _sample:
            sample[col] = copy.copy(_sample[col])

        if self.resampler:
            sample = self.resampler(sample)
        return sample

    def __len__(self):
        return len(self.ut)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
