from torch.utils.data import DataLoader

from loader.global_loader import GlobalLoader
from set.base_dataset import BaseDataset
from task.base_batch import BaseBatch
from task.base_task import BaseTask


class BaseDataLoader(DataLoader):
    def __init__(self, dataset: BaseDataset, task: BaseTask, **kwargs):
        super().__init__(
            dataset=dataset,
            **kwargs
        )

        self.auto_dataset = dataset
        self.task = task

    # def start_epoch(self, current_epoch, total_epoch):
    #     self.task.start_epoch(current_epoch, total_epoch)
    #     return self

    def test(self):
        self.task.test()
        return self

    def eval(self):
        self.task.eval()
        return self

    def train(self):
        self.task.train()
        return self

    def __iter__(self):
        iterator = super().__iter__()

        while True:
            try:
                batch = next(iterator)
                batch = self.task.rebuild_batch(self, batch)  # type: BaseBatch
                yield batch
            except StopIteration:
                return

    @classmethod
    def get_loader(cls, loader: GlobalLoader, mode):
        shuffle = loader.data.split[mode].shuffle  # NONE, FALSE, TRUE
        if shuffle not in [True, False]:  # CAN NOT USE "IF SHUFFLE"
            shuffle = loader.data.shuffle or False

        return cls(
            dataset=loader.datasets[mode],
            task=loader.primary_task,
            shuffle=shuffle,
            batch_size=loader.exp.policy.batch_size,
            pin_memory=loader.exp.policy.pin_memory,
        )
