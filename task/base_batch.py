import torch

from typing import Optional, Dict


class BaseBatch:
    def __init__(self, batch):
        from task.base_task import BaseTask

        self.append = batch['append']  # type: Dict[str, any]
        self.inputs = batch['inputs']  # type: Dict[str, torch.Tensor]
        self.task = None  # type: Optional[BaseTask]

    def dict(self):
        return self.__dict__


class SeqBatch(BaseBatch):
    pass
