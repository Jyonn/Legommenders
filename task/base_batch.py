import torch

from typing import Optional, Dict

from loader.global_setting import Setting


class BaseBatch:
    def __init__(self, batch):
        from task.base_task import BaseTask

        self.append = batch['append']  # type: Dict[str, any]
        self.inputs = batch['inputs']  # type: Dict[str, torch.Tensor]
        self.batch_size = len(list(self.append.values())[0])
        self.task = None  # type: Optional[BaseTask]

    def dict(self):
        return self.__dict__


class SeqBatch(BaseBatch):
    def __init__(self, batch):
        super().__init__(batch)

        self.attention_mask = self.get_attention_mask()

    def get_attention_mask(self) -> torch.Tensor:
        mask = None
        for col in self.inputs:
            seq = self.inputs[col]
            if not mask:
                mask = torch.zeros(*seq.shape)
            col_mask = (seq > Setting.PAD).long()
            mask |= col_mask
        return mask


class HSeqBatch(BaseBatch):
    def __init__(self, batch):
        super().__init__(batch)
        self.doc_clicks = batch['doc_clicks']  # type: Dict[str, torch.Tensor]
        self.doc_candidates = batch['doc_candidates']  # type: Dict[str, torch.Tensor]
