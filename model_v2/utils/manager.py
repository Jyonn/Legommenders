import random

import torch
from tqdm import tqdm

from model_v2.recommenders.base_neg_recommender import BaseNegRecommender
from model_v2.recommenders.base_recommender import BaseRecommender, BaseRecommenderConfig
from model_v2.utils.nr_depot import NRDepot
from set.base_dataset import BaseDataset
from utils.stacker import Stacker
from utils.timer import Timer


class Status:
    def __init__(self):
        self.is_training = True
        self.is_evaluating = False
        self.is_testing = False

    def train(self):
        self.is_training = True
        self.is_evaluating = False
        self.is_testing = False

    def eval(self):
        self.is_training = False
        self.is_evaluating = True
        self.is_testing = False

    def test(self):
        self.is_training = False
        self.is_evaluating = False
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
        self.neg_col = self.column_map.neg_col
        self.user_col = self.column_map.user_col
        self.clicks_mask_col = self.column_map.clicks_mask_col

        # document manager and cache
        self.doc_dataset = None
        self.doc_inputer = None
        self.doc_cache = None
        self.stacker = Stacker(aggregator=torch.stack)
        if self.use_content:
            self.doc_dataset = BaseDataset(nrd=doc_nrd)
            self.doc_inputer = recommender.news_encoder.inputer
            self.doc_cache = self.get_doc_cache()

        self.user_cache = dict()

        # clicks
        self.user_inputer = recommender.user_encoder.inputer
        self.max_click_num = self.user_inputer.depot.get_max_length(self.clicks_col)

        # negative sampling
        self.use_neg_sampling = recommender.use_neg_sampling
        self.news_size = doc_nrd.depot.get_vocab_size(self.candidate_col)

    def get_doc_cache(self):
        doc_cache = []
        for sample in tqdm(self.doc_dataset):
            doc_cache.append(self.doc_inputer.sample_rebuilder(sample))
        return doc_cache

    def rebuild_sample(self, sample):
        # reform features
        len_clicks = len(sample[self.clicks_col])
        sample[self.clicks_mask_col] = [1] * len_clicks + [0] * (self.max_click_num - len_clicks)
        if self.use_content:
            sample[self.clicks_col].extend([0] * (self.max_click_num - len_clicks))
        sample[self.candidate_col] = [sample[self.candidate_col]]

        # negative sampling
        if self.use_neg_sampling:
            assert isinstance(self.recommender, BaseNegRecommender)
            if not self.status.is_testing:
                true_negs = sample[self.neg_col]
                rand_neg = max(self.recommender.neg_count - len(true_negs), 0)
                neg_samples = random.sample(true_negs, k=min(self.recommender.neg_count, len(true_negs)))
                neg_samples += [random.randint(0, self.news_size - 1) for _ in range(rand_neg)]
                sample[self.candidate_col].extend(neg_samples)
        del sample[self.neg_col]

        # content injection and tensorization
        if self.use_content:
            sample[self.candidate_col] = self.stacker([self.doc_cache[nid] for nid in sample[self.candidate_col]])
            if sample[self.user_col] in self.user_cache:
                sample[self.clicks_col] = self.user_cache[sample[self.user_col]]
            else:
                sample[self.clicks_col] = self.stacker([self.doc_cache[nid] for nid in sample[self.clicks_col]])
                self.user_cache[sample[self.user_col]] = sample[self.clicks_col]
        else:
            sample[self.candidate_col] = torch.tensor(sample[self.candidate_col], dtype=torch.long)
            sample[self.clicks_col] = self.user_inputer.sample_rebuilder(sample)

        sample[self.clicks_mask_col] = torch.tensor(sample[self.clicks_mask_col], dtype=torch.long)
        return sample
