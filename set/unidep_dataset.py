from UniTok import UniDep
from torch.utils.data import Dataset

from utils.splitter import Splitter


class UniDepDataset(Dataset):
    def __init__(
            self,
            depot: UniDep,
            splitter: Splitter = None,
            mode=None,
    ):
        self.depot = depot
        self.mode = mode
        self.sample_size = self.depot.sample_size

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

    def pack_sample(self, index):
        raise NotImplementedError
