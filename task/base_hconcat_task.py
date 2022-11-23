from abc import ABC

import torch
from tqdm import tqdm

from loader.depot.vocab_loader import VocabLoader
from loader.embedding.embedding_init import EmbeddingInit
from loader.global_setting import Setting
from set.base_dataset import BaseDataset
from task.base_batch import BaseBatch
from task.base_concat_task import BaseConcatTask
from task.base_doc_seq_task import BaseDocSeqTask
from task.utils.padding import Padding
from utils.structure import Structure


class BaseHConcatTask(BaseConcatTask, BaseDocSeqTask, ABC):
    batcher = BaseBatch

    def __init__(
            self,
            dataset: BaseDataset,
            label_col='label',
            clicks_col='history',
            candidate_col='nid',
            user_col='uid',
            **kwargs
    ):
        super().__init__(dataset, **kwargs)

        self.label_col = label_col
        self.user_col = user_col
        self.clicks_col = clicks_col
        self.candidate_col = candidate_col
        self.max_click_num = self.depot.get_max_length(self.clicks_col)

        self.doc_padding = Padding(
            depot=self.doc_depot,
            order=self.doc_order
        )
        self.padding = Padding(
            depot=self.dataset.depot,
            order=self.dataset.order,
        )
        self.doc_cache = self.get_doc_cache()
        self.click_cache = dict()

    def get_doc_cache(self):
        doc_cache = []
        for sample in tqdm(self.doc_dataset):
            doc_cache.append(self.doc_padding(sample['inputs']))
        return doc_cache

    def get_doc_clicks(self, sample: dict):
        user_id = sample['append'][self.user_col]
        if user_id in self.click_cache:
            return self.click_cache[user_id]

        clicks = sample['inputs'][self.clicks_col]
        doc_clicks = self.doc_parser(clicks)
        doc_clicks.extend([self.doc_padding.sample_wise_pad()] * (self.max_click_num - len(doc_clicks)))
        self.click_cache[user_id] = doc_clicks
        return doc_clicks

    def rebuild_sample(self, sample: dict, dataset: BaseDataset):
        inputs = sample['inputs']
        candidates = [inputs[self.candidate_col]]

        doc_clicks = self.get_doc_clicks(sample)
        doc_candidates = self.doc_parser(candidates)

        inputs[self.clicks_col] = self.stacker(doc_clicks)
        inputs[self.candidate_col] = self.stacker(doc_candidates)
        return sample

    def _get_embedding(
            self,
            inputs,
            input_embeddings: list,
            batch_size: int,
            embedding_init: EmbeddingInit,
            vocab_loader: VocabLoader,
    ):
        table = embedding_init.get_table()
        for col in inputs:
            col_input = inputs[col]
            if isinstance(col_input, torch.Tensor):
                seq = col_input.to(Setting.device)  # type: torch.Tensor # [B, L], [B]
                mask = (seq > Setting.UNSET).long()  # type: torch.Tensor  # [B, L], [B]
                seq *= mask

                vocab = vocab_loader[col].name
                embedding = table[vocab](seq)
                embedding = embedding.reshape(batch_size, -1, embedding_init.hidden_size)
                mask = mask.reshape(batch_size, -1, 1)
                embedding *= mask
                embedding = embedding.sum(dim=1) / mask.sum(dim=1)  # [B, D]
                input_embeddings.append(embedding)
            else:
                assert isinstance(col_input, dict)
                self._get_embedding(
                    inputs=col_input,
                    input_embeddings=input_embeddings,
                    batch_size=batch_size,
                    embedding_init=embedding_init,
                    vocab_loader=vocab_loader,
                )

    def get_embeddings(
            self,
            batch: BaseBatch,
            embedding_init: EmbeddingInit,
            vocab_loader: VocabLoader
    ):
        input_embeddings = []

        self._get_embedding(
            inputs=batch.inputs,
            input_embeddings=input_embeddings,
            batch_size=batch.batch_size,
            embedding_init=embedding_init,
            vocab_loader=vocab_loader,
        )

        input_embeddings = torch.cat(input_embeddings, dim=-1)
        return input_embeddings

    def doc_parser(self, l: list):
        samples = []
        for doc_id in l:
            # sample = self.doc_dataset[doc_id]
            # samples.append(self.doc_padding(sample['inputs']))
            samples.append(self.doc_cache[doc_id])
        return samples
