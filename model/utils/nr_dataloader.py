from torch.utils.data import DataLoader

from model.utils.manager import Manager
from loader.base_dataset import BaseDataset


class NRDataLoader(DataLoader):
    dataset: BaseDataset

    def __init__(self, manager: Manager, **kwargs):
        super().__init__(
            **kwargs
        )

        self.manager = manager

    def test(self):
        self.manager.status.test()
        self.manager.recommender.end_fast_eval()
        self.manager.recommender.start_fast_eval(self.manager.doc_cache)
        if self.manager.recommender.user_plugin:
            self.manager.recommender.user_plugin.end_fast_eval()
            self.manager.recommender.user_plugin.start_fast_eval()
        return self

    def eval(self):
        self.manager.status.eval()
        self.manager.recommender.end_fast_eval()
        self.manager.recommender.start_fast_eval(self.manager.doc_cache)
        if self.manager.recommender.user_plugin:
            self.manager.recommender.user_plugin.end_fast_eval()
            self.manager.recommender.user_plugin.start_fast_eval()
        return self

    def train(self):
        self.manager.status.train()
        self.manager.recommender.end_fast_eval()
        if self.manager.recommender.user_plugin:
            self.manager.recommender.user_plugin.end_fast_eval()
        return self

    # def __iter__(self):
    #     iterator = super().__iter__()
    #
    #     while True:
    #         try:
    #             batch = next(iterator)
    #             batch = self.task.rebuild_batch(self, batch)  # type: BaseBatch
    #             yield batch
    #         except StopIteration:
    #             return
