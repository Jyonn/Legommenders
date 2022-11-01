from abc import ABC
from typing import Dict

import torch

from loader.depot.vocab_loader import VocabLoader
from loader.embedding.embedding_init import EmbeddingInit
from loader.global_setting import Setting
from task.base_batch import BaseBatch
from task.base_task import BaseTask


class BaseConcatTask(BaseTask, ABC):

    def get_embeddings(
            self,
            batch: BaseBatch,
            embedding_init: EmbeddingInit,
            vocab_loader: VocabLoader
    ):
        inputs = batch.inputs  # type: Dict[str, torch.Tensor]
        table = embedding_init.get_table()
        input_embeddings = []

        for col in inputs:
            seq = inputs[col].to(Setting.device)  # type: torch.Tensor # [B, L], [B]
            mask = (seq > Setting.UNSET).long()  # type: torch.Tensor  # [B, L], [B]
            seq *= mask

            vocab = vocab_loader[col].name
            embedding = table[vocab](seq)
            embedding = embedding.reshape(batch.batch_size, -1, embedding_init.hidden_size)
            mask = mask.reshape(batch.batch_size, -1, 1)
            embedding *= mask
            embedding = embedding.sum(dim=1) / mask.sum(dim=1)  # [B, D]
            input_embeddings.append(embedding)

        return torch.concat(input_embeddings)
