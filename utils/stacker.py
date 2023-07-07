import copy
from collections import OrderedDict
from typing import List, Callable

import torch


class Stacker:
    def __init__(self, aggregator: Callable = None):
        self.aggregator = aggregator or torch.stack

    def _build_prototype(self, item: dict):
        # prototype = OrderedDict()
        prototype = dict()
        for k in item.keys():
            if isinstance(item[k], dict):
                prototype[k] = self._build_prototype(item[k])
            else:
                prototype[k] = []
        return prototype

    def _insert_data(self, prototype: dict, item: dict):
        for k in item.keys():
            if isinstance(item[k], dict):
                self._insert_data(prototype[k], item[k])
            else:
                prototype[k].append(item[k])

    def _aggregate(self, prototype: dict, apply: Callable = None):
        for k in prototype.keys():
            if isinstance(prototype[k], dict):
                self._aggregate(prototype[k])
            else:
                prototype[k] = self.aggregator(prototype[k])
                if apply:
                    prototype[k] = apply(prototype[k])

    def stack(self, item_list: List[dict], apply: Callable = None):
        prototype = self._build_prototype(item_list[0])
        for item in item_list:
            self._insert_data(prototype, item)
        if self.aggregator:
            self._aggregate(prototype, apply=apply)
        return prototype

    def __call__(self, item_list: List[dict], apply: Callable = None):
        return self.stack(item_list, apply=apply)


class OneDepthStacker(Stacker):
    def __init__(self, aggregator: Callable = None):
        super().__init__(aggregator)

    def _build_prototype(self, item: dict):
        prototype = OrderedDict()
        for k in item.keys():
            prototype[k] = []
        return prototype


class FastStacker(Stacker):
    def __init__(self, aggregator: Callable = None):
        super().__init__(aggregator)
        self.prototype = None

    def stack(self, item_list: List[dict]):
        if not self.prototype:
            self.prototype = self._build_prototype(item_list[0])
        prototype = copy.deepcopy(self.prototype)
        for item in item_list:
            self._insert_data(prototype, item)
        if self.aggregator:
            self._aggregate(prototype)
        return prototype


if __name__ == '__main__':
    a = dict(
        append=OrderedDict(),
        inputs=dict(
            title=[1, 2, 3]
        )
    )
    b = dict(
        append=OrderedDict(),
        inputs=dict(
            title=[3, 2, 1]
        )
    )
    stacker = Stacker(torch.tensor)
    print(stacker.stack([a, b]))
