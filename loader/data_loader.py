from torch.utils.data import DataLoader as BaseDataLoader

from loader.resampler import Resampler
from loader.data_set import DataSet


class DataLoader(BaseDataLoader):
    dataset: DataSet

    def __init__(self, resampler: Resampler, user_set: DataSet, **kwargs):
        super().__init__(**kwargs)

        self.resampler = resampler
        self.user_set = user_set

        self.cacher = resampler.legommender.cacher

    def test(self):
        self.resampler.legommender.eval()
        self.resampler.status.test()
        self.cacher.cache(
            item_contents=self.resampler.item_cache,
            user_contents=self.user_set,
        )
        return self

    def eval(self):
        self.resampler.legommender.eval()
        self.resampler.status.eval()
        self.cacher.cache(
            item_contents=self.resampler.item_cache,
            user_contents=self.user_set,
        )
        return self

    def train(self):
        self.resampler.legommender.train()
        self.resampler.status.train()
        self.cacher.clean()
        return self
