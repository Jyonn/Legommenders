from collections import OrderedDict
from typing import List, Callable

import torch


class Stacker:
    def __init__(self, aggregator: Callable = None):
        self.aggregator = aggregator or torch.stack

    def _build_prototype(self, item: dict):
        prototype = OrderedDict()
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

    def _aggregate(self, prototype: dict):
        for k in prototype.keys():
            if isinstance(prototype[k], dict):
                self._aggregate(prototype[k])
            else:
                prototype[k] = self.aggregator(prototype[k])

    def stack(self, item_list: List[dict]):
        prototype = self._build_prototype(item_list[0])
        for item in item_list:
            self._insert_data(prototype, item)
        if self.aggregator:
            self._aggregate(prototype)
        return prototype

    def __call__(self, item_list: List[dict]):
        return self.stack(item_list)


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
