from typing import List

from torch import nn

from loader.embedding.embedding_init import EmbeddingInit
from task.base_task import BaseTask

TASK_LIST = []  # type: List[BaseTask]
TASKS = {task.name: task for task in TASK_LIST}


class TaskLoader:
    def __init__(self, exp, dataset):
        self.exp = exp
        self.dataset = dataset

        self.tasks = []

    def parse(self):
        for task_info in self.exp.tasks:
            if task_info.name not in TASKS:
                raise ValueError(f'No matched task: {task_info.name}')

            task_class = TASKS[task_info.name]
            params = task_info.params.dict()

            task = task_class(dataset=self.dataset, **params)
            self.tasks.append(task)

        return self.tasks

    def register_task_tokens(self, embedding_init: EmbeddingInit):
        for task in self.tasks:  # type: BaseTask
            embedding_init.register_vocab(task.vocab_name, task.task_tokens.get_size())

    def get_primary_task(self):
        if len(self.tasks) == 1:
            return self.tasks[0]

    def get_task_modules(self):
        table = dict()
        for task in self.tasks:  # type: BaseTask
            table[task.name] = task.get_module()
        return nn.ModuleDict(table)
