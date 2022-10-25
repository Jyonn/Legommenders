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
