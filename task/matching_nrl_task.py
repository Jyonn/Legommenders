from typing import Dict

import torch
from torch import nn

from loader.depot.vocab_loader import VocabLoader
from loader.embedding.embedding_init import EmbeddingInit
from loader.global_setting import Setting
from set.base_dataset import BaseDataset
from task.base_batch import NRLBatch
from task.base_loss import BaseLoss
from task.base_neg_task import BaseNegTask
from utils.structure import Structure


# from task.base_seq_task import BaseSeqTask


class MatchingNRLTask(BaseNegTask):
    batcher = NRLBatch
    name = 'matching-nrl'
    dynamic_loader = True

    def __init__(
            self,
            dataset: BaseDataset,
            label_col='label',
            clicks_col='history',
            candidate_col='nid',
            **kwargs,
    ):
        super().__init__(dataset, **kwargs)

        self.label_col = label_col
        self.clicks_col = clicks_col
        self.candidate_col = candidate_col
        self.max_click_num = self.depot.get_max_length(self.clicks_col)

        self.criterion = nn.CrossEntropyLoss()

    def rebuild_sample(self, sample: dict, dataset: BaseDataset):
        candidates = [sample['append'][self.candidate_col]]
        candidates.extend(self.negative_sampling(sample, self.depot.get_vocab_size(self.candidate_col)))

        clicks = sample['inputs'][self.clicks_col]
        click_mask = torch.tensor([1] * len(clicks) + [0] * (self.max_click_num - len(clicks)), dtype=torch.long)
        clicks.extend([clicks[-1]] * (self.max_click_num - len(clicks)))

        sample['inputs'][self.clicks_col] = torch.tensor(clicks, dtype=torch.long)
        sample['inputs'][self.candidate_col] = torch.tensor(candidates, dtype=torch.long)
        sample['click_mask'] = click_mask

        # sample = super(MatchingNRLTask, self).rebuild_sample(sample, dataset)
        return sample

    def get_embeddings(
            self,
            batch: NRLBatch,
            embedding_init: EmbeddingInit,
            vocab_loader: VocabLoader,
    ):
        inputs = batch.inputs
        vocab = vocab_loader[self.clicks_col].name
        table = embedding_init.get_table()
        embedding = table[vocab]

        def _get_embeddings(col):
            seq = inputs[col].to(Setting.device)  # type: torch.Tensor # [B, L], [B]
            return embedding(seq)

        return _get_embeddings(self.clicks_col), _get_embeddings(self.candidate_col), batch.click_mask

    def calculate_loss(self, output, batch: NRLBatch, **kwargs) -> BaseLoss:
        label = torch.zeros(batch.batch_size, dtype=torch.long).to(Setting.device)
        return BaseLoss(self.criterion(output, label))

    def calculate_scores(self, output, batch: NRLBatch, **kwargs):
        return output.squeeze(dim=-1).detach().cpu().tolist()
