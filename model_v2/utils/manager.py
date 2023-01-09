import random
from typing import Union

import torch
from UniTok import UniDep
from tqdm import tqdm

from loader.depot.depot_cache import DepotCache
from model_v2.recommenders.base_model import BaseRecommender, BaseRecommenderConfig
from model_v2.utils.nr_depot import NRDepot
from set.base_dataset import BaseDataset
from utils.stacker import Stacker


class DepotToDatasetManager:
    def __init__(
            self,
            depot: NRDepot,
    ):
        self.depot = depot

        self.dataset = BaseDataset(
            nrd=self.depot,
            order=self.order,
            append=self.append,
        )


class Status:
    def __init__(self):
        self.is_training = True
        self.is_validating = False
        self.is_testing = False

    def train(self):
        self.is_training = True
        self.is_validating = False
        self.is_testing = False

    def validate(self):
        self.is_training = False
        self.is_validating = True
        self.is_testing = False

    def test(self):
        self.is_training = False
        self.is_validating = False
        self.is_testing = True


class Manager:
    def __init__(
            self,
            recommender: BaseRecommender,
            doc_nrd: NRDepot,
    ):
        self.status = Status()

        # parameter assignment
        self.recommender = recommender
        self.config = recommender.config  # type: BaseRecommenderConfig
        self.use_content = self.config.use_news_content

        self.column_map = recommender.column_map
        self.clicks_col = self.column_map.clicks_col
        self.candidate_col = self.column_map.candidate_col
        self.label_col = self.column_map.label_col
        self.clicks_mask_col = self.column_map.clicks_mask_col

        # document manager
        self.doc_dataset = None
        self.doc_cache = None
        self.doc_inputer = None
        self.stacker = Stacker(aggregator=torch.stack)
        if self.use_content:
            self.doc_dataset = BaseDataset(nrd=doc_nrd)
            self.doc_cache = self.get_doc_cache()
            self.doc_inputer = recommender.news_encoder.inputer

        # clicks
        self.max_click_num = recommender.user_encoder.inputer.depot.get_max_length(self.clicks_col)

        # negative sampling
        self.neg_col = None
        self.neg_count = None
        self.use_neg_sampling = recommender.user_encoder.use_neg_sampling
        if self.use_neg_sampling:
            self.neg_col = recommender.user_encoder.config.negative_sampling.neg_col
            self.neg_count = recommender.user_encoder.config.negative_sampling.neg_count
        self.news_size = recommender.news_encoder.inputer.depot.get_vocab_size(self.candidate_col)

    def get_doc_cache(self):
        doc_cache = []
        for sample in tqdm(self.doc_dataset):
            doc_cache.append(self.doc_inputer.sample_rebuilder(sample))
        return doc_cache

    def rebuild_sample(self, sample):
        # reform features
        len_clicks = len(sample[self.clicks_col])
        sample[self.clicks_mask_col] = [1] * len_clicks + [0] * (self.max_click_num - len_clicks)
        sample[self.clicks_col].extend([0] * (self.max_click_num - len_clicks))
        sample[self.candidate_col] = [sample[self.candidate_col]]

        # negative sampling
        if self.use_neg_sampling:
            if not self.status.is_testing:
                rand_neg = self.neg_count
                neg_samples = []
                if self.neg_col:
                    true_negs = sample[self.neg_col]
                    rand_neg = max(self.neg_count - len(true_negs), 0)
                    neg_samples = random.sample(true_negs, k=min(self.neg_count, len(true_negs)))
                neg_samples += [random.randint(0, self.news_size - 1) for _ in range(rand_neg)]
                sample[self.candidate_col].extend(neg_samples)
            if self.neg_col:
                del sample[self.neg_col]

        # content injection and tensorization
        for col in [self.candidate_col, self.clicks_col]:
            if self.use_content:
                sample[col] = self.stacker([self.doc_cache[nid] for nid in sample[col]])
            else:
                sample[col] = torch.tensor(sample[col], dtype=torch.long)

        sample[self.clicks_mask_col] = torch.tensor(sample[self.clicks_mask_col], dtype=torch.long)
        return sample

    def rebuild_batch(self, batch):
        # for concat inputer
        # clicks col (no content): [batch_size, max_click_num]
        # clicks col (with content): [batch_size, max_click_num, max_seq_len]
        # clicks mask col: [batch_size, max_click_num]
        # candidate col (no content): [batch_size, neg_count + 1] or [batch_size, 1]
        # candidate col (with content): [batch_size, neg_count + 1, max_seq_len] or [batch_size, 1, max_seq_len]

        # for avg inputer
        # clicks col (no content): [batch_size, max_click_num]
        # clicks col (with content): [batch_size, max_click_num, max_seq_len]
        pass

