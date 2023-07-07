from typing import Callable

import torch
from tqdm import tqdm

from loader.global_setting import Setting


class TorchPager:
    def __init__(self, contents: list, model: Callable, page_size: int, features: list, **kwargs):
        self.contents = contents
        self.model = model
        self.page_size = page_size
        self.features = features
        self.caches = {feature: [] for feature in features}
        self.current = {feature: [] for feature in features}
        self.current_count = self.cache_count = 0

    def get_features(self, content, index) -> dict:
        raise NotImplementedError

    def push_to_cache(self):
        for feature, value in self.current.items():
            self.caches[feature].append(value)
            self.current[feature] = []
        self.current_count = 0
        self.cache_count += 1

    def prepare(self):
        for index, content in enumerate(tqdm(self.contents)):
            features = self.get_features(content, index=index)
            for feature, value in features.items():
                self.current[feature].append(value)

            self.current_count += 1

            if self.current_count == self.page_size:
                self.push_to_cache()

        if self.current_count:
            self.push_to_cache()

    def combine(self, slices, features, output):
        raise NotImplementedError

    def stack_features(self, index):
        return {
            feature: torch.stack(self.caches[feature][index]).to(Setting.device)
            for feature in self.features
        }

    def process(self):
        with torch.no_grad():
            for i in tqdm(range(self.cache_count)):
                current_page_size = len(self.caches[self.features[0]][i])
                features = self.stack_features(i)
                output = self.model(**features)
                slices = slice(i * self.page_size, i * self.page_size + current_page_size)
                self.combine(slices, features, output)
