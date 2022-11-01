from abc import ABC

import torch
from UniTok import UniDep

from set.base_dataset import BaseDataset
from task.base_task import BaseTask
from utils.stacker import Stacker


class BaseDocSeqTask(BaseTask, ABC):
    def __init__(
            self,
            dataset: BaseDataset,
            doc_depot,
            doc_order=None,
            **kwargs
    ):
        super().__init__(dataset, **kwargs)
        self.doc_depot = UniDep(doc_depot)
        self.doc_order = doc_order or ['title']

        self.doc_dataset = BaseDataset(
            depot=self.doc_depot,
            order=self.doc_order,
            append=[],
        )

        for col in self.doc_order:
            self.add_vocab(self.doc_depot.vocab_depot[self.doc_depot.get_vocab(col)], col=col)

        self.stacker = Stacker(aggregator=torch.stack)

    def doc_parser(self, l: list):
        raise NotImplementedError
