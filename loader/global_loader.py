import os
from typing import Dict

from loader.depot.depot_loader import FilterUniDep
from loader.embedding.embedding_init import EmbeddingInit
from loader.global_setting import Setting
from set.base_dataset import BaseDataset
from task.task_loader import TaskLoader
from utils.printer import printer
from utils.splitter import Splitter


class GlobalLoader:
    def __init__(self, data, model, exp):
        Setting.global_loader = self
        self.data = data
        self.model = model
        self.exp = exp
        self.print = printer.DATA_Cblue_

        self.depots, self.splitter = FilterUniDep.parse(self.data)  # type: Dict[FilterUniDep], Splitter

        self.embedding_init = EmbeddingInit.parse(self.data, self.model, self.a_depot)
        self.datasets = BaseDataset.parse(self.data, self.depots, self.splitter)

        self.train_set = self.datasets.get(Setting.TRAIN)
        self.dev_set = self.datasets.get(Setting.DEV)
        self.test_set = self.datasets.get(Setting.TEST)

        self.task_loader = TaskLoader(self.exp, self.a_set)
        self.tasks = self.task_loader.parse()
        self.task_loader.register_task_tokens(self.embedding_init)
        self.primary_task = self.task_loader.get_primary_task()

    @property
    def a_depot(self):
        return list(self.depots.items())[0]

    @property
    def a_set(self):
        return self.train_set or self.dev_set or self.test_set
