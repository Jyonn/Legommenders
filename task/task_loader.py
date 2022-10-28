from typing import List, Type

from UniTok import Vocab
from oba import Obj
from torch import nn

from loader.depot.vocab_loader import VocabLoader
from loader.embedding.embedding_init import EmbeddingInit
from task.base_task import BaseTask
from task.matching_task import MatchingTask

TASK_LIST = [
    MatchingTask,
]  # type: List[Type[BaseTask]]
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
            params = dict()
            if task_info.params:
                params = Obj.raw(task_info.params)

            task = task_class(dataset=self.dataset, **params)
            self.tasks.append(task)

        return self.tasks

    def register_task_vocabs(self, embedding_init: EmbeddingInit, vocab_loader: VocabLoader):
        for task in self.tasks:  # type: BaseTask
            vocabs = task.vocabs
            if vocabs:
                for vocab in vocabs:
                    col, vocab = vocab
                    embedding_init.register_vocab(vocab)
                    vocab_loader.register(vocab, col=col)

    def get_primary_task(self):
        if len(self.tasks) == 1:
            return self.tasks[0]

    def get_task_modules(self):
        table = dict()
        for task in self.tasks:  # type: BaseTask
            table[task.name] = task.get_module()
        return nn.ModuleDict(table)
