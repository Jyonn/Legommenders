from task.base_batch import BaseBatch
from task.base_hseq_task import BaseHSeqTask
from task.base_loss import BaseLoss


class MatchingTask(BaseHSeqTask):
    name = 'matching'

    def calculate_loss(self, output, batch: BaseBatch, **kwargs) -> BaseLoss:
        pass
