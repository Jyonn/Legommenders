import torch

from typing import Optional, Dict

from loader.global_setting import Setting


class BaseBatch:
    def __init__(self, batch):
        from task.base_task import BaseTask

        self.append = batch['append']  # type: Dict[str, any]
        self.inputs = batch['inputs']  # type: Dict[str, torch.Tensor]
        self.batch_size = len(list(self.inputs.values())[0])
        self.task = None  # type: Optional[BaseTask]

    def dict(self):
        return self.__dict__


class SeqBatch(BaseBatch):
    def __init__(self, batch):
        super().__init__(batch)

        self.attention_mask = batch['attention_mask']


class HSeqBatch(BaseBatch):
    def __init__(self, batch):
        super().__init__(batch)
        self.doc_clicks = SeqBatch(batch['doc_clicks'])
        self.doc_candidates = SeqBatch(batch['doc_candidates'])
