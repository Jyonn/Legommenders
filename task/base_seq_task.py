from abc import ABC
from typing import Dict

import torch

from loader.depot.vocab_loader import VocabLoader
from loader.embedding.embedding_init import EmbeddingInit
from loader.global_setting import Setting
from set.base_dataset import BaseDataset
from task.base_batch import SeqBatch
from task.base_task import BaseTask
from task.utils.sequencer import Sequencer


class BaseSeqTask(BaseTask, ABC):
    batcher = SeqBatch

    def __init__(
            self,
            dataset: BaseDataset,
            use_cls_token=False,
            use_sep_token=False,
    ):
        super().__init__(dataset)
        self.sequencer = Sequencer(
            depot=self.dataset.depot,
            order=self.dataset.order,
            use_cls_token=use_cls_token,
            use_sep_token=use_sep_token,
        )
        self.add_vocab(self.sequencer.vocab)

    def rebuild_sample(self, sample: dict, dataset: BaseDataset):
        sample['inputs'], sample['attention_mask'] = self.sequencer(sample['inputs'])
        return sample

    @staticmethod
    def _get_embedding(
            inputs: Dict[str, torch.Tensor],
            embedding_init: EmbeddingInit,
            vocab_loader: VocabLoader,
    ):
        shape = list(inputs.values())[0].shape
        input_embeddings = torch.zeros(
            *shape,
            embedding_init.hidden_size,
            dtype=torch.float
        ).to(Setting.device)
        table = embedding_init.get_table()

        for col in inputs:
            seq = inputs[col]  # [B, L]
            mask = (seq > Setting.UNSET).long()  # type: torch.Tensor  # [B, L]
            seq *= mask

            vocab = vocab_loader[col].name
            embedding = table[vocab](seq)
            embedding *= mask.unsqueeze(-1)

            input_embeddings += embedding
        return input_embeddings

    def get_embeddings(
            self,
            batch: SeqBatch,
            embedding_init: EmbeddingInit,
            vocab_loader: VocabLoader,
    ):
        inputs = batch.inputs
        return self._get_embedding(
            inputs=inputs,
            embedding_init=embedding_init,
            vocab_loader=vocab_loader,
        )
