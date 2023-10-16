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
        self.cacher = manager.recommender.cacher
        self.plugin = manager.recommender.user_plugin

    def test(self):
        self.manager.status.test()
        self.cacher.end_caching_doc_repr()
        self.cacher.start_caching_doc_repr(self.manager.item_cache)
        if self.plugin:
            self.plugin.end_fast_eval()
            self.plugin.start_fast_eval()
        self.cacher.end_caching_user_repr()
        self.cacher.start_caching_user_repr(self.manager.user_dataset)
        return self

    def eval(self):
        self.manager.status.eval()
        self.cacher.end_caching_doc_repr()
        self.cacher.start_caching_doc_repr(self.manager.item_cache)
        if self.plugin:
            self.plugin.end_fast_eval()
            self.plugin.start_fast_eval()
        self.cacher.end_caching_user_repr()
        self.cacher.start_caching_user_repr(self.manager.user_dataset)
        return self

    def train(self):
        self.manager.status.train()
        self.cacher.end_caching_doc_repr()
        if self.plugin:
            self.plugin.end_fast_eval()
        self.cacher.end_caching_user_repr()
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
