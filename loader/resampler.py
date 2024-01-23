import random

import numpy as np
import torch
from tqdm import tqdm

from loader.meta import Meta
from loader.status import Status
from model.legommender import Legommender, LegommenderConfig
from loader.data_hub import DataHub
from loader.data_set import DataSet
from utils.stacker import Stacker, FastStacker
from utils.timer import Timer


class Resampler:
    def __init__(
            self,
            legommender: Legommender,
            item_hub: DataHub,
            status: Status,
    ):
        self.status = status
        self.timer = Timer(activate=True)

        # parameter assignment
        self.legommender = legommender
        self.config = legommender.config  # type: LegommenderConfig
        self.use_item_content = self.config.use_item_content

        self.column_map = legommender.column_map
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
        self.stacker = FastStacker(aggregator=torch.stack)
        # self.stacker = default_collate
        if self.use_item_content:
            self.item_dataset = DataSet(hub=item_hub)
            self.item_inputer = legommender.item_encoder.inputer
            self.item_cache = self._build_item_cache()

        self.user_cache = dict()
        self.candidate_cache = dict()

        # clicks
        self.user_inputer = legommender.user_encoder.inputer
        self.max_click_num = self.user_inputer.depot.get_max_length(self.clicks_col)

        # negative sampling
        self.use_neg_sampling = legommender.use_neg_sampling
        self.item_size = item_hub.depot.get_vocab_size(self.candidate_col)

    def _build_item_cache(self):
        item_cache = []
        for sample in tqdm(self.item_dataset):
            item_cache.append(self.item_inputer.sample_rebuilder(sample))
        return item_cache

    @staticmethod
    def pack_tensor(array):
        return torch.tensor(array, dtype=torch.long)

    def rebuild_candidates(self, sample):
        # convert candidate to list
        if not isinstance(sample[self.candidate_col], list):
            sample[self.candidate_col] = [sample[self.candidate_col]]

        # negative sampling
        if self.use_neg_sampling:
            if self.status.is_training or (self.status.is_evaluating and Meta.simple_dev):
                # During testing or non-simple-dev evaluation,
                # the legommender will directly calculate the scores.
                # Therefore, we don't need to do negative sampling for cross-entropy loss.
                true_negs = sample[self.neg_col] if self.neg_col else []
                rand_count = max(self.legommender.neg_count - len(true_negs), 0)

                neg_samples = random.sample(true_negs, k=min(self.legommender.neg_count, len(true_negs)))
                neg_samples += [random.randint(0, self.item_size - 1) for _ in range(rand_count)]
                sample[self.candidate_col].extend(neg_samples)
        if self.neg_col:
            del sample[self.neg_col]

        if not self.use_item_content:
            # if not using item content, we don't need to rebuild candidate contents
            sample[self.candidate_col] = self.pack_tensor(sample[self.candidate_col])
            return

        if self.legommender.llm_skip:
            # if llm_skip, we don't need to rebuild candidate contents,
            # as llm cache has stored their content knowledge
            sample[self.candidate_col] = self.pack_tensor(sample[self.candidate_col])
            return
        if self.legommender.cacher.item.cached:
            # if item cached, we don't need to rebuild candidate contents,
            # as item cache has stored their content knowledge
            sample[self.candidate_col] = self.pack_tensor(sample[self.candidate_col])
            return

        # start to inject content knowledge
        # if self.use_neg_sampling:
            # when using negative sampling, we need to rebuild candidate contents
        sample[self.candidate_col] = self.stacker([self.item_cache[nid] for nid in sample[self.candidate_col]])
        return

        # when not using negative sampling, we can use cache to speed up
        if sample[self.candidate_col][0] not in self.candidate_cache:
            item_id = sample[self.candidate_col][0]
            sample[self.candidate_col] = self.stacker([self.item_cache[nid] for nid in sample[self.candidate_col]])
            self.candidate_cache[item_id] = sample[self.candidate_col]
        else:
            sample[self.candidate_col] = self.candidate_cache[sample[self.candidate_col][0]]

    def rebuild_clicks(self, sample):
        if self.legommender.cacher.user.cached:
            # if user cached, we don't need to rebuild clicks,
            # as user cache has stored their click knowledge
            del sample[self.clicks_col]
            return

        # convert clicks to list
        if isinstance(sample[self.clicks_col], np.ndarray):
            sample[self.clicks_col] = sample[self.clicks_col].tolist()
        len_clicks = len(sample[self.clicks_col])
        # padding clicks
        sample[self.clicks_mask_col] = [1] * len_clicks + [0] * (self.max_click_num - len_clicks)
        sample[self.clicks_mask_col] = torch.tensor(sample[self.clicks_mask_col], dtype=torch.long)
        if self.use_item_content:
            sample[self.clicks_col].extend([0] * (self.max_click_num - len_clicks))

        if not self.use_item_content:
            # if not using item content, we use vanilla inputer provided by user operator to rebuild clicks
            sample[self.clicks_col] = self.user_inputer.sample_rebuilder(sample)
            return
        if self.legommender.flatten_mode:
            # in flatten mode, click contents will be rebuilt by user inputer
            sample[self.clicks_col] = self.user_inputer.sample_rebuilder(sample)
            sample[self.clicks_mask_col] = self.user_inputer.get_mask(sample[self.clicks_col])
            return

        if self.legommender.llm_skip:
            # if llm_skip, we don't need to rebuild click contents,
            # as llm cache has stored their content knowledge
            sample[self.clicks_col] = self.pack_tensor(sample[self.clicks_col])
            return
        if self.legommender.cacher.item.cached:
            # if item cached, we don't need to rebuild candidate contents,
            # as item cache has stored their content knowledge
            sample[self.clicks_col] = self.pack_tensor(sample[self.clicks_col])
            return

        # we can cache click content knowledge to speed up
        if sample[self.user_col] in self.user_cache:
            sample[self.clicks_col] = self.user_cache[sample[self.user_col]]
        else:
            sample[self.clicks_col] = self.stacker([self.item_cache[nid] for nid in sample[self.clicks_col]])
            self.user_cache[sample[self.user_col]] = sample[self.clicks_col]

    def rebuild(self, sample):
        """transform sample to tensor-based dict"""

        self.rebuild_candidates(sample)
        self.rebuild_clicks(sample)
        return sample

    def __call__(self, sample):
        return self.rebuild(sample)
