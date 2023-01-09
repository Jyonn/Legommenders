from torch.utils.data import DataLoader

from model_v2.utils.manager import Manager
from set.base_dataset import BaseDataset


class NRDataLoader(DataLoader):
    dataset: BaseDataset

    def __init__(self, manager: Manager, **kwargs):
        super().__init__(
            **kwargs
        )

        self.manager = manager

    def test(self):
        self.manager.status.test()
        return self

    def eval(self):
        self.manager.status.eval()
        return self

    def train(self):
        self.manager.status.train()
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
