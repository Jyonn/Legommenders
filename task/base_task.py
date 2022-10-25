from typing import Type, OrderedDict, List

from UniTok import Vocab

from loader.depot.vocab_loader import VocabLoader
from loader.embedding.embedding_init import EmbeddingInit
from loader.global_setting import Setting
from set.base_dataset import BaseDataset
from task.base_batch import BaseBatch
from task.base_loss import BaseLoss
from utils.printer import printer, Color


class BaseTask:
    name: str
    batcher: Type[BaseBatch]
    dynamic_loader: bool = False

    def __init__(
            self,
            dataset: BaseDataset,
            **kwargs,
    ):
        self.dataset = dataset
        self.depot = dataset.depot

        self.print = printer[(self.__class__.__name__, '-', Color.YELLOW)]
        self.module = None
        self.vocabs: List[Vocab] = []

    @classmethod
    def pad(cls, l: list, max_len: int):
        return l + [Setting.PAD] * (max_len - len(l))

    def rebuild_sample(self, sample: dict):
        inputs = sample['inputs']
        for col in inputs:
            max_len = self.depot.get_max_length(col)
            if max_len:
                inputs[col] = self.pad(inputs[col], max_len)
        return sample

    def rebuild_batch(self, dataloader, batch):
        batch = self.batcher(batch)
        return self._rebuild_batch(dataloader, batch)

    def _rebuild_batch(self, dataloader, batch: BaseBatch):
        return batch

    def get_module(self):
        if self.module:
            return self.module
        return self._get_module()

    def _get_module(self):
        pass

    @property
    def vocab_name(self):
        return f'__TASK_{self.name}'

    def get_task_specific_vocab(self):
        return Vocab(name=self.vocab_name)

    # model training and testing
    def get_embeddings(self, batch: BaseBatch, embedding_init: EmbeddingInit, vocab_loader: VocabLoader):
        raise NotImplementedError

    def rebuild_output(self, output, batch: BaseBatch):
        return output

    def calculate_loss(self, output, batch: BaseBatch, **kwargs) -> BaseLoss:
        raise NotImplementedError
