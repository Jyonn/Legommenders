import pandas as pd
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
        self.test_results = {}

    def calculate_loss(self, output, batch: HSeqBatch, **kwargs) -> BaseLoss:
        label = torch.zeros(batch.batch_size, dtype=torch.long).to(Setting.device)
        return BaseLoss(self.criterion(output, label))

    def test(self):
        super(MatchingTask, self).test()
        self.test_results = dict(
            score=[],
            click=[],
            imp=[],
        )

    def on_test(self, output: torch.Tensor, batch: HSeqBatch, **kwargs):
        score = output.squeeze(-1).detach().cpu().tolist()
        imp = batch.append['imp'].tolist()
        click = batch.append['click'].tolist()
        self.test_results['score'].extend(score)
        self.test_results['imp'].extend(imp)
        self.test_results['click'].extend(click)

    def summarize_test(self, **kwargs):
        test_results = pd.DataFrame(self.test_results)
        test_results.groupby('imp')