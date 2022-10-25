from task.base_batch import BaseBatch
from task.base_loss import BaseLoss
from task.base_seq_task import BaseSeqTask


class MatchingTask(BaseSeqTask):
    name = 'matching'

    def calculate_loss(self, output, batch: BaseBatch, **kwargs) -> BaseLoss:
        pass
