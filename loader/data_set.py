import copy

from torch.utils.data import Dataset as BaseDataset

from loader.data_hub import DataHub
from utils.timer import Timer


class DataSet(BaseDataset):
    def __init__(
            self,
            hub: DataHub,
            resampler=None,
    ):
        self.hub = hub
        self.ut = hub.ut
        self.order = hub.order
        self.append = hub.append
        self.all_cols = [*self.order, *self.append]

        self.resampler = resampler

        self.sample_size = len(self.ut)
        self.split_range = (0, self.sample_size)

        self.timer = Timer(activate=True)

    def __getitem__(self, index):
        index += self.split_range[0]
        return self.pack_sample(index)

    def __len__(self):
        mode_range = self.split_range
        return mode_range[1] - mode_range[0]

    def pack_sample(self, index):
        _sample = self.ut[index]
        sample = dict()
        for col in [*self.order, *self.append]:
            # value = _sample[col]
            # if isinstance(value, np.ndarray):
            #     value = value.tolist()
            # sample[col] = copy.copy(value)
            sample[col] = copy.copy(_sample[col])

        if self.resampler:
            sample = self.resampler(sample)
        return sample
        #
        # sample = {col: copy.copy(sample[col]) for col in self.all_cols}
        # if self.resampler:
        #     sample = self.resampler(sample)
        # return sample

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
