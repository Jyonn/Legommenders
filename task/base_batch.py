import torch

from typing import Optional, Dict


class BaseBatch:
    def __init__(self, batch):
        from task.base_task import BaseTask

        self.append = batch['append']  # type: Dict[str, any]
        self.inputs = batch['inputs']  # type: Dict[str, torch.Tensor]

        # self.batch_size = len(list(self.append.values())[0])
        # if not self.batch_size:
        #     raise ValueError('unable to parse batch size')
        self.batch_size = self.parse_batch_size(self.inputs)

        self.task = None  # type: Optional[BaseTask]

    def parse_batch_size(self, d: dict):
        for k in d:
            if isinstance(d[k], dict):
                batch_size = self.parse_batch_size(d[k])
                if batch_size:
                    return batch_size
            elif isinstance(d[k], torch.Tensor):
                return d[k].shape[0]

    def dict(self):
        return self.__dict__


class SeqBatch(BaseBatch):
    def __init__(self, batch):
        super().__init__(batch)

        self.attention_mask = batch['attention_mask']


class NRLBatch(BaseBatch):
    def __init__(self, batch):
        super().__init__(batch)
        self.click_mask = batch['click_mask']  # type: torch.Tensor


class HSeqBatch(BaseBatch):
    def __init__(self, batch):
        super().__init__(batch)
        self.doc_clicks = SeqBatch(batch['doc_clicks'])
        self.doc_candidates = SeqBatch(batch['doc_candidates'])
        self.click_mask = batch['click_mask']  # type: torch.Tensor
