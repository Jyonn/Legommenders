from abc import ABC

import torch
from tqdm import tqdm

from loader.depot.vocab_loader import VocabLoader
from loader.embedding.embedding_init import EmbeddingInit
from set.base_dataset import BaseDataset
from task.base_batch import HSeqBatch
from task.base_doc_seq_task import BaseDocSeqTask
from task.base_neg_task import BaseNegTask
from task.base_seq_task import BaseSeqTask
from task.utils.sequencer import Sequencer


class BaseHSeqTask(BaseSeqTask, BaseDocSeqTask, BaseNegTask, ABC):
    """
    Base Hierarchical Sequence Task

    - including document encoding
    """

    batcher = HSeqBatch

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

        self.doc_cache = self.get_doc_cache()

    def get_doc_cache(self):
        doc_sequencer = Sequencer(
            depot=self.doc_depot,
            order=self.doc_order,
            use_cls_token=False,
            use_sep_token=False,
        )

        doc_cache = []
        for sample in tqdm(self.doc_dataset):
            sample['inputs'], sample['attention_mask'] = doc_sequencer(sample['inputs'])
            doc_cache.append(sample)
        return doc_cache

    def rebuild_sample(self, sample: dict, dataset: BaseDataset):
        candidates = [sample['append'][self.candidate_col]]
        candidates.extend(self.negative_sampling(sample, self.depot.get_vocab_size(self.candidate_col)))
        doc_candidates = self.doc_parser(candidates)

        clicks = sample['inputs'][self.clicks_col]
        click_mask = torch.tensor([1] * len(clicks) + [0] * (self.max_click_num - len(clicks)), dtype=torch.long)
        doc_clicks = self.doc_parser(clicks)
        doc_clicks.extend([doc_clicks[-1]] * (self.max_click_num - len(doc_clicks)))

        sample['doc_clicks'] = self.stacker(doc_clicks)
        sample['doc_candidates'] = self.stacker(doc_candidates)
        sample['click_mask'] = click_mask
        sample['append']['_candidates'] = candidates
        sample = super(BaseHSeqTask, self).rebuild_sample(sample, dataset)
        return sample

    def get_embeddings(
            self,
            batch: HSeqBatch,
            embedding_init: EmbeddingInit,
            vocab_loader: VocabLoader,
    ):
        clicks_embedding = self._get_embedding(
            inputs=batch.doc_clicks.inputs,
            embedding_init=embedding_init,
            vocab_loader=vocab_loader,
        )  # [B, N, L, D]
        candidates_embedding = self._get_embedding(
            inputs=batch.doc_candidates.inputs,
            embedding_init=embedding_init,
            vocab_loader=vocab_loader,
        )
        return clicks_embedding, \
            candidates_embedding, \
            batch.doc_clicks.attention_mask, \
            batch.click_mask, \
            batch.doc_candidates.attention_mask

    def doc_parser(self, l: list):
        samples = []
        for doc_id in l:
            samples.append(self.doc_cache[doc_id])
        return samples
