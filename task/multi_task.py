import random
from typing import OrderedDict, List, Optional

from set.base_dataset import BaseDataset
from task.base_task import BaseTask


class MultiTask(BaseTask):
    def __init__(
            self,
            dataset: BaseDataset,
            tasks: List[BaseTask],
    ):
        super().__init__(dataset)

        self.tasks = tasks
        self.present_task = self.choose_task()  # type: BaseTask
        self.sample_static_rebuilder = None
        self.sample_dynamic_rebuilder = self.dynamic_rebuild_sample

    def choose_task(self):
        return random.choice(self.tasks)

    def static_rebuild_sample(self, sample: OrderedDict):
        pass

    def dynamic_rebuild_sample(self, sample: OrderedDict):
        rebuilder = self.present_task.sample_static_rebuilder or self.present_task.sample_dynamic_rebuilder
        if rebuilder:
            sample = rebuilder(sample)
        return sample

    def rebuild_batch(self, dataloader, batch):
        pass

    def _get_module(self):
        pass
