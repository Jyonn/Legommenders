from typing import Dict

from loader.base_dataloader import BaseDataLoader
from loader.depot.depot_loader import FilterUniDep
from loader.depot.vocab_loader import VocabLoader
from loader.embedding.embedding_init import EmbeddingInit
from loader.global_setting import Setting
from model import model_loader
from model.model_container import ModelContainer
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
        self.vocab_loader = VocabLoader()
        self.vocab_loader.load_from_depot(self.a_depot, self.data.order)

        self.embedding_init = EmbeddingInit.parse(self.data, self.model, self.a_depot)
        self.datasets = BaseDataset.parse(self.data, self.depots, self.splitter)

        self.train_set = self.datasets.get(Setting.TRAIN)
        self.dev_set = self.datasets.get(Setting.DEV)
        self.test_set = self.datasets.get(Setting.TEST)

        self.task_loader = TaskLoader(self.exp, self.a_set)
        self.tasks = self.task_loader.parse()
        self.task_loader.register_task_vocabs(self.embedding_init, self.vocab_loader)
        self.primary_task = self.task_loader.get_primary_task()

        for dataset in self.datasets.values():
            dataset.register_task(self.primary_task)

        self.core_model = model_loader.parse(self.model)
        self.model_container = ModelContainer(
            task_loader=self.task_loader,
            embedding_init=self.embedding_init,
            vocab_loader=self.vocab_loader,
            model=self.core_model,
        )

    @property
    def a_depot(self):
        return list(self.depots.values())[0]

    @property
    def a_set(self):
        return self.train_set or self.dev_set or self.test_set

    def get_dataloader(self, mode):
        shuffle = self.data.split[mode].shuffle  # NONE, FALSE, TRUE
        if shuffle not in [True, False]:  # CAN NOT USE "IF SHUFFLE"
            shuffle = self.data.shuffle or False

        return BaseDataLoader(
            dataset=self.datasets[mode],
            task=self.primary_task,
            shuffle=shuffle,
            batch_size=self.exp.policy.batch_size,
            pin_memory=self.exp.policy.pin_memory,
        )
