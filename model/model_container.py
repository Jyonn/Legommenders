import torch
from torch import nn

from loader.depot.vocab_loader import VocabLoader
from loader.embedding.embedding_init import EmbeddingInit
from loader.global_setting import Setting
from task.base_batch import BaseBatch
from task.base_task import BaseTask
from task.task_loader import TaskLoader


class ModelContainer(nn.Module):
    def __init__(
            self,
            task_loader: TaskLoader,
            embedding_init: EmbeddingInit,
            vocab_loader: VocabLoader,
            model: nn.Module,
    ):
        super().__init__()
        self.task_loader = task_loader
        self.embedding_init = embedding_init
        self.vocab_loader = vocab_loader
        self.model = model

        self.task_modules = task_loader.get_task_modules()
        self.embedding_table = embedding_init.get_table()

    def forward(self, task: BaseTask, batch: BaseBatch):
        input_embeddings = task.get_embeddings(
            batch=batch,
            embedding_init=self.embedding_init,
            vocab_loader=self.vocab_loader,
        )
        if not isinstance(input_embeddings, tuple):
            input_embeddings = (input_embeddings, )
        input_embeddings = [embed.to(Setting.device)
                            if isinstance(embed, torch.Tensor) else embed
                            for embed in input_embeddings]
        outputs = self.model(*input_embeddings)
        outputs = task.rebuild_output(outputs, batch)
        return outputs
