from torch import nn

from loader.depot.vocab_loader import VocabLoader
from loader.embedding.embedding_init import EmbeddingInit
from loader.global_setting import Setting
from set.base_dataset import BaseDataset
from task.base_batch import BaseBatch
from task.base_hconcat_task import BaseHConcatTask
from task.base_loss import BaseLoss


class RankingTask(BaseHConcatTask):
    name = 'ranking'
    dynamic_loader = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.criterion = nn.BCELoss()

    def calculate_loss(self, output, batch: BaseBatch, **kwargs) -> BaseLoss:
        labels = batch.append[self.label_col].to(Setting.device).float()
        scores = output.squeeze(dim=-1)
        return BaseLoss(self.criterion(scores, labels))

    def calculate_scores(self, output, batch: BaseBatch, **kwargs):
        return output.squeeze(dim=-1).detach().cpu().tolist()
