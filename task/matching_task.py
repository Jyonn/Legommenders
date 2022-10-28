import torch
from torch import nn

from loader.global_setting import Setting
from task.base_batch import HSeqBatch
from task.base_hseq_task import BaseHSeqTask
from task.base_loss import BaseLoss


class MatchingTask(BaseHSeqTask):
    name = 'matching'
    dynamic_loader = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.criterion = nn.CrossEntropyLoss()

    def calculate_loss(self, output, batch: HSeqBatch, **kwargs) -> BaseLoss:
        label = torch.zeros(batch.batch_size, dtype=torch.long).to(Setting.device)
        return self.criterion(output, label)
