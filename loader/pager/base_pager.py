from typing import Callable

import torch

from loader.env import Env
from utils import bars


class BasePager:
    def __init__(self, contents: list, model: Callable, page_size: int, desc: str = None, **kwargs):
        self.contents = contents
        self.model = model
        self.page_size = page_size
        self.current = dict()
        self.current_count = self.cache_count = 0
        self.desc = desc or 'Pager Caching'

    def get_features(self, content, index) -> dict:
        raise NotImplementedError

    def run(self):
        for index, content in enumerate(bars.DescBar(desc=self.desc)(self.contents)):
            features = self.get_features(content, index=index)
            for feature, value in features.items():
                if feature not in self.current:
                    self.current[feature] = []
                self.current[feature].append(value)

            self.current_count += 1

            if self.current_count == self.page_size:
                self._process()
        if self.current_count:
            self._process()

    def combine(self, slices, features, output):
        raise NotImplementedError

    def stack_features(self):
        return {
            feature: torch.stack(self.current[feature]).to(Env.device)
            for feature in self.current
        }

    def _process(self):
        features = self.stack_features()

        output = self.model(**features)
        slices = slice(self.cache_count * self.page_size, self.cache_count * self.page_size + self.current_count)
        self.combine(slices, features, output)

        self.cache_count += 1
        self.current_count = 0
        for feature in self.current:
            self.current[feature] = []
