from loader.data_hubs import DataHubs
from loader.data_set import DataSet
from loader.meta import Phases
from loader.resampler import Resampler


class DataSets:
    def __init__(self, hubs: DataHubs, resampler: Resampler):
        self.hubs = hubs

        self.train_set = self.dev_set = self.test_set = None
        if hubs.train_hub:
            self.train_set = DataSet(hub=self.hubs.train_hub, resampler=resampler)
        if hubs.dev_hub:
            self.dev_set = DataSet(hub=self.hubs.dev_hub, resampler=resampler)
        if hubs.test_hub:
            self.test_set = DataSet(hub=self.hubs.test_hub, resampler=resampler)
        self.user_set = DataSet(hub=self.hubs.fast_eval_hub, resampler=resampler)

        self.sets = {
            Phases.train: self.train_set,
            Phases.dev: self.dev_set,
            Phases.test: self.test_set,
            Phases.fast_eval: self.user_set,
        }

    def __getitem__(self, item):
        return self.sets[item]

    def a_set(self):
        return self.train_set or self.dev_set or self.test_set
