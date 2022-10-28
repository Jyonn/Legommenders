import random
from abc import ABC

import torch
from UniTok import UniDep

from loader.depot.vocab_loader import VocabLoader
from loader.embedding.embedding_init import EmbeddingInit
from set.base_dataset import BaseDataset
from task.base_batch import HSeqBatch
from task.base_loss import BaseLoss
from task.base_seq_task import BaseSeqTask
from task.utils.sequencer import Sequencer
from utils.stacker import Stacker


class BaseHSeqTask(BaseSeqTask, ABC):
    """
    Base Hierarchical Sequence Task

    - including document encoding
    """

    batcher = HSeqBatch

    def __init__(
            self,
            dataset: BaseDataset,
            doc_depot,
            doc_order=None,
            label_col='label',
            clicks_col='history',
            candidate_col='nid',
            neg_count=4,
    ):
        super().__init__(dataset)

        self.doc_depot = UniDep(doc_depot)
        self.doc_order = doc_order or ['title']
        self.label_col = label_col
        self.clicks_col = clicks_col
        self.candidate_col = candidate_col
        self.max_click_num = self.depot.get_max_length(self.clicks_col)

        self.neg_count = neg_count
        self.neg_index = list(range(self.doc_depot.sample_size))

        self.doc_dataset = BaseDataset(
            depot=self.doc_depot,
            order=self.doc_order,
            append=[],
        )

        self.doc_sequencer = Sequencer(
            depot=self.doc_depot,
            order=self.doc_order,
            use_cls_token=False,
            use_sep_token=False,
        )

        for col in self.doc_order:
            self.add_vocab(self.doc_depot.vocab_depot[self.doc_depot.get_vocab(col)], col=col)

        self.stacker = Stacker(aggregator=torch.stack)

    def negative_sampling(self):
        random.shuffle(self.neg_index)
        return self.neg_index[:self.neg_count]

    def doc_parser(self, l: list):
        samples = []
        for doc_id in l:
            sample = self.doc_dataset[doc_id]
            sample['inputs'], sample['attention_mask'] = self.doc_sequencer(sample['inputs'])
            samples.append(sample)
        return samples

    def rebuild_sample(self, sample: dict, dataset: BaseDataset):
        inputs = sample['inputs']
        append = sample['append']
        clicks = inputs[self.clicks_col]
        candidates = [append[self.candidate_col]]
        if not self.is_testing:
            candidates.extend(self.negative_sampling())

        if not clicks:
            print(sample)
        doc_clicks = self.doc_parser(clicks)
        doc_candidates = self.doc_parser(candidates)
        doc_clicks.extend([doc_clicks[-1]] * (self.max_click_num - len(doc_clicks)))

        sample['doc_clicks'] = self.stacker(doc_clicks)
        sample['doc_candidates'] = self.stacker(doc_candidates)
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
        )
        candidates_embedding = self._get_embedding(
            inputs=batch.doc_candidates.inputs,
            embedding_init=embedding_init,
            vocab_loader=vocab_loader,
        )
        return clicks_embedding, candidates_embedding
