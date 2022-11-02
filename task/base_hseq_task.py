import random
from abc import ABC

import torch
from UniTok import UniDep
from tqdm import tqdm

from loader.depot.vocab_loader import VocabLoader
from loader.embedding.embedding_init import EmbeddingInit
from set.base_dataset import BaseDataset
from task.base_batch import HSeqBatch
from task.base_doc_seq_task import BaseDocSeqTask
from task.base_seq_task import BaseSeqTask
from task.utils.sequencer import Sequencer


class BaseHSeqTask(BaseSeqTask, BaseDocSeqTask, ABC):
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
            neg_count=4,
            neg_col='neg',
            **kwargs,
    ):
        super().__init__(dataset, **kwargs)

        self.label_col = label_col
        self.clicks_col = clicks_col
        self.candidate_col = candidate_col
        self.max_click_num = self.depot.get_max_length(self.clicks_col)

        self.neg_count = neg_count
        self.neg_col = neg_col
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

    def negative_sampling(self, sample: dict):
        neg_samples = []
        if not self.is_testing:
            rand_neg = self.neg_count
            neg_samples = []
            if self.neg_col and self.neg_col in sample['append']:
                true_negs = sample['append'][self.neg_col]
                rand_neg = max(self.neg_count - len(true_negs), 0)
                neg_samples = random.choices(true_negs, k=min(self.neg_count, len(true_negs)))
            neg_samples += [random.randint(0, self.doc_depot.sample_size - 1) for _ in range(rand_neg)]
        if self.neg_col and self.neg_col in sample['append']:
            del sample['append'][self.neg_col]
        return neg_samples

    def rebuild_sample(self, sample: dict, dataset: BaseDataset):
        clicks = sample['inputs'][self.clicks_col]
        candidates = [sample['append'][self.candidate_col]]
        candidates.extend(self.negative_sampling(sample))

        doc_clicks = self.doc_parser(clicks)
        doc_candidates = self.doc_parser(candidates)
        click_mask = torch.tensor([1] * len(doc_clicks) + [0] * (self.max_click_num - len(doc_clicks)), dtype=torch.long)
        doc_clicks.extend([doc_clicks[-1]] * (self.max_click_num - len(doc_clicks)))

        # for col in append:
        #     append[col] = torch.tensor(append[col])

        sample['doc_clicks'] = self.stacker(doc_clicks)
        sample['doc_candidates'] = self.stacker(doc_candidates)
        sample['click_mask'] = click_mask
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
            # sample = self.doc_dataset[doc_id]
            # sample['inputs'], sample['attention_mask'] = self.doc_sequencer(sample['inputs'])
            # samples.append(sample)
            samples.append(self.doc_cache[doc_id])
        return samples
