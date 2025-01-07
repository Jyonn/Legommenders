import random

import torch

from loader.env import Env
from loader.data_set import DataSet
from model.lego_config import LegoConfig
from utils import bars
from utils.stacker import FastStacker
from utils.timer import Timer


class Resampler:
    def __init__(
            self,
            lego_config: LegoConfig,
    ):
        self.timer = Timer(activate=True)

        # parameter assignment
        self.lego_config = lego_config  # type: LegoConfig
        self.use_item_content = self.lego_config.use_item_content

        self.column_map = self.lego_config.column_map
        self.history_col = self.column_map.history_col
        self.item_col = self.column_map.item_col
        self.click_col = self.column_map.label_col
        self.user_col = self.column_map.user_col
        self.neg_col = self.column_map.neg_col
        self.mask_col = self.column_map.mask_col

        # item manager and cache
        self.item_dataset = None
        self.item_inputer = None
        self.item_cache = None
        self.stacker = FastStacker(aggregator=torch.stack)
        if self.use_item_content:
            self.item_dataset = DataSet(ut=self.lego_config.item_ut)
            self.item_inputer = self.lego_config.item_operator.inputer
            self.item_cache = self._build_item_cache()

        self.user_cache = dict()

        # clicks
        self.user_inputer = self.lego_config.user_operator.inputer
        self.max_click_num = self.user_inputer.ut.meta.jobs[self.history_col].max_len

        # negative sampling
        self.use_neg_sampling = self.lego_config.use_neg_sampling
        self.item_size = self.lego_config.item_ut.meta.jobs[self.item_col].tokenizer.vocab.size

    def _build_item_cache(self):
        item_cache = []
        for sample in (bars.DescBar(desc='Building Item Cache'))(self.item_dataset):
            item_cache.append(self.item_inputer(sample))
        return item_cache

    @staticmethod
    def pack_tensor(array):
        return torch.tensor(array, dtype=torch.long)

    def rebuild_candidates(self, sample):
        if not isinstance(sample[self.item_col], list):
            sample[self.item_col] = [sample[self.item_col]]

        # negative sampling
        if self.use_neg_sampling:
            if Env.is_training or (Env.is_evaluating and Env.simple_dev):
                # During testing or non-simple-dev evaluation,
                # the legommender will directly calculate the scores.
                # Therefore, we don't need to do negative sampling for cross-entropy loss.
                true_negs = sample[self.neg_col] if self.neg_col else []
                rand_count = max(self.lego_config.neg_count - len(true_negs), 0)

                neg_samples = random.sample(true_negs, k=min(self.lego_config.neg_count, len(true_negs)))
                neg_samples += [random.randint(0, self.item_size - 1) for _ in range(rand_count)]
                sample[self.item_col].extend(neg_samples)

        if self.neg_col:
            del sample[self.neg_col]

        if not self.use_item_content:
            # if not using item content, we don't need to rebuild candidate contents
            sample[self.item_col] = self.pack_tensor(sample[self.item_col])
            return

        if Env.lm_cache:
            # if llm_cache, we don't need to rebuild candidate contents,
            # as llm cache has stored their content knowledge
            sample[self.item_col] = self.pack_tensor(sample[self.item_col])
            return
        if Env.item_cache:
            # if item cached, we don't need to rebuild candidate contents,
            # as item cache has stored their content knowledge
            sample[self.item_col] = self.pack_tensor(sample[self.item_col])
            return

        # start to inject content knowledge
        # if self.use_neg_sampling:
            # when using negative sampling, we need to rebuild candidate contents
        sample[self.item_col] = self.stacker([self.item_cache[nid] for nid in sample[self.item_col]])
        return
        #
        # # when not using negative sampling, we can use cache to speed up
        # if sample[self.candidate_col][0] not in self.candidate_cache:
        #     item_id = sample[self.candidate_col][0]
        #     sample[self.candidate_col] = self.stacker([self.item_cache[nid] for nid in sample[self.candidate_col]])
        #     self.candidate_cache[item_id] = sample[self.candidate_col]
        # else:
        #     sample[self.candidate_col] = self.candidate_cache[sample[self.candidate_col][0]]

    def rebuild_clicks(self, sample):
        if Env.user_cache:
            # if user cached, we don't need to rebuild clicks,
            # as user cache has stored their click knowledge
            del sample[self.history_col]
            return

        # convert clicks to list
        len_clicks = len(sample[self.history_col])
        # padding clicks
        sample[self.mask_col] = [1] * len_clicks + [0] * (self.max_click_num - len_clicks)
        sample[self.mask_col] = torch.tensor(sample[self.mask_col], dtype=torch.long)
        if self.use_item_content:
            sample[self.history_col].extend([0] * (self.max_click_num - len_clicks))

        if not self.use_item_content:
            # if not using item content, we use vanilla inputer provided by user operator to rebuild clicks
            sample[self.history_col] = self.user_inputer(sample)
            return
        if self.lego_config.user_operator_class.flatten_mode:
            # in flatten mode, click contents will be rebuilt by user inputer
            sample[self.history_col] = self.user_inputer(sample)
            sample[self.mask_col] = self.user_inputer.get_mask(sample[self.history_col])
            return

        if Env.lm_cache:
            # if llm_cache, we don't need to rebuild click contents,
            # as llm cache has stored their content knowledge
            sample[self.history_col] = self.pack_tensor(sample[self.history_col])
            return
        if Env.item_cache:
            # if item cached, we don't need to rebuild candidate contents,
            # as item cache has stored their content knowledge
            sample[self.history_col] = self.pack_tensor(sample[self.history_col])
            return

        # we can cache click content knowledge to speed up
        if sample[self.user_col] in self.user_cache:
            sample[self.history_col] = self.user_cache[sample[self.user_col]]
        else:
            sample[self.history_col] = self.stacker([self.item_cache[nid] for nid in sample[self.history_col]])
            self.user_cache[sample[self.user_col]] = sample[self.history_col]

    def rebuild(self, sample):
        """transform sample to tensor-based dict"""

        self.rebuild_candidates(sample)
        self.rebuild_clicks(sample)
        return sample

    def __call__(self, sample):
        return self.rebuild(sample)
