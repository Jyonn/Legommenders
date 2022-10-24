from typing import OrderedDict

from set.base_dataset import BaseDataset
from task.base_task import BaseTask


class MatchingTask(BaseTask):
    def __init__(self, dataset: BaseDataset):
        super(MatchingTask, self).__init__(dataset=dataset)

    def static_rebuild_sample(self, sample: OrderedDict):
        pass

    def dynamic_rebuild_sample(self, sample: OrderedDict):
        pass

    def rebuild_batch(self, dataloader, batch):
        pass

    def _get_module(self):
        pass
