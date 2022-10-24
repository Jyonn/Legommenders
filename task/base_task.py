from typing import Optional, Type, Callable, OrderedDict

from UniTok import Vocab

from set.base_dataset import BaseDataset
from task.base_batch import BaseBatch
from utils.printer import printer, Color


class BaseTask:
    name: str
    batcher: Type[BaseBatch]
    sample_static_rebuilder: Optional[Callable] = None
    sample_dynamic_rebuilder: Optional[Callable] = None

    def __init__(
            self,
            dataset: BaseDataset
    ):
        self.dataset = dataset

        self.print = printer[(self.__class__.__name__, '-', Color.YELLOW)]
        self.task_tokens = Vocab(name='task')
        self._module = None

    def static_rebuild_sample(self, sample: OrderedDict):
        raise NotImplementedError

    def dynamic_rebuild_sample(self, sample: OrderedDict):
        raise NotImplementedError

    def rebuild_batch(self, dataloader, batch):
        raise NotImplementedError

    def get_module(self):
        if self._module:
            return self._module
        return self._get_module()

    def _get_module(self):
        raise NotImplementedError

    @property
    def vocab_name(self):
        return f'__TASK_{self.name}'
