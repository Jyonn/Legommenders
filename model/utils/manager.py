import random

import numpy as np
import torch
from tqdm import tqdm

from loader.global_setting import Setting
# from model.recommenders.base_neg_recommender import BaseNegRecommender
from model.recommenders.base_recommender import BaseRecommender, BaseRecommenderConfig
from model.utils.nr_depot import DataHub
from loader.base_dataset import BaseDataset
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
            doc_nrd: DataHub,
            user_nrd: DataHub,
    ):
        self.status = Status()

        self.timer = Timer(activate=True)

        # parameter assignment
        self.recommender = recommender
        self.config = recommender.config  # type: BaseRecommenderConfig
        self.use_news_content = self.config.use_item_content

        self.column_map = recommender.column_map
        self.clicks_col = self.column_map.clicks_col
        self.candidate_col = self.column_map.candidate_col
        self.label_col = self.column_map.label_col
        self.neg_col = self.column_map.neg_col
        self.user_col = self.column_map.user_col
        self.index_col = self.column_map.index_col
        self.clicks_mask_col = self.column_map.clicks_mask_col

        # document manager and cache
        self.item_dataset = None
        self.item_inputer = None
        self.item_cache = None
        self.stacker = Stacker(aggregator=torch.stack)
        # self.stacker = default_collate
        if self.use_news_content:
            self.item_dataset = BaseDataset(nrd=doc_nrd)
            self.item_inputer = recommender.item_encoder.inputer
            self.item_cache = self.get_doc_cache()

        self.user_cache = dict()
        self.candidate_cache = dict()

        # clicks
        self.user_inputer = recommender.user_encoder.inputer
        self.max_click_num = self.user_inputer.depot.get_max_length(self.clicks_col)

        # negative sampling
        self.use_neg_sampling = recommender.use_neg_sampling
        self.news_size = doc_nrd.depot.get_vocab_size(self.candidate_col)

        # user manager
        self.user_dataset = BaseDataset(nrd=user_nrd, manager=self)

    def get_doc_cache(self):
        doc_cache = []
        for sample in tqdm(self.item_dataset):
            doc_cache.append(self.item_inputer.sample_rebuilder(sample))
        return doc_cache

    def rebuild_sample(self, sample):
        # self.timer.run('rebuild 1')
        # reform features
        if isinstance(sample[self.clicks_col], np.ndarray):
            sample[self.clicks_col] = sample[self.clicks_col].tolist()
        len_clicks = len(sample[self.clicks_col])
        sample[self.clicks_mask_col] = [1] * len_clicks + [0] * (self.max_click_num - len_clicks)
        if self.use_news_content:
            sample[self.clicks_col].extend([0] * (self.max_click_num - len_clicks))
        if not isinstance(sample[self.candidate_col], list):
            sample[self.candidate_col] = [sample[self.candidate_col]]
        # self.timer.run('rebuild 1')

        # self.timer.run('rebuild 2')
        # negative sampling
        if self.use_neg_sampling:
            # assert isinstance(self.recommender, BaseNegRecommender)
            # if not self.status.is_testing:
            if self.status.is_training or (self.status.is_evaluating and Setting.simple_dev):
                if self.neg_col:
                    true_negs = sample[self.neg_col]
                else:
                    true_negs = []
                rand_neg = max(self.recommender.neg_count - len(true_negs), 0)
                neg_samples = random.sample(true_negs, k=min(self.recommender.neg_count, len(true_negs)))
                neg_samples += [random.randint(0, self.news_size - 1) for _ in range(rand_neg)]
                sample[self.candidate_col].extend(neg_samples)
        if self.neg_col:
            del sample[self.neg_col]
        # self.timer.run('rebuild 2')
        #
        # self.timer.run('rebuild 3')
        # content injection and tensorization
        if self.use_news_content and not self.recommender.llm_skip and not self.recommender.cacher.fast_doc_eval:
            if self.use_neg_sampling or sample[self.candidate_col][0] not in self.candidate_cache:
                stacked_doc = self.stacker([self.item_cache[nid] for nid in sample[self.candidate_col]])
            else:
                stacked_doc = self.candidate_cache[sample[self.candidate_col][0]]
            sample[self.candidate_col] = stacked_doc

            if sample[self.user_col] in self.user_cache:
                sample[self.clicks_col] = self.user_cache[sample[self.user_col]]
            else:
                sample[self.clicks_col] = self.stacker([self.item_cache[nid] for nid in sample[self.clicks_col]])
                self.user_cache[sample[self.user_col]] = sample[self.clicks_col]
        else:
            sample[self.candidate_col] = torch.tensor(sample[self.candidate_col], dtype=torch.long)
            if self.recommender.llm_skip or self.recommender.cacher.fast_doc_eval:
                sample[self.clicks_col] = torch.tensor(sample[self.clicks_col], dtype=torch.long)
            else:
                sample[self.clicks_col] = self.user_inputer.sample_rebuilder(sample)

        sample[self.clicks_mask_col] = torch.tensor(sample[self.clicks_mask_col], dtype=torch.long)
        # self.timer.run('rebuild 3')

        return sample
