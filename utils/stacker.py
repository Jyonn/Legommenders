import copy
from collections import OrderedDict
from typing import List, Callable, Union

import torch


class Stacker:
    def __init__(self, aggregator: Callable = None):
        self.aggregator = aggregator or torch.stack

    def _build_prototype(self, item) -> Union[dict, list]:
        # prototype = OrderedDict()
        if not isinstance(item, dict):
            return []

        prototype = dict()
        for k in item.keys():
            if isinstance(item[k], dict):
                prototype[k] = self._build_prototype(item[k])
            else:
                prototype[k] = []
        return prototype

    def _insert_data(self, prototype: Union[dict, list], item):
        if not isinstance(item, dict):
            prototype.append(item)
            return

        for k in item.keys():
            if isinstance(item[k], dict):
                self._insert_data(prototype[k], item[k])
            else:
                prototype[k].append(item[k])

    def _aggregate(self, prototype: Union[dict, list], apply: Callable = None):
        if isinstance(prototype, list):
            prototype = self.aggregator(prototype)
            if apply:
                prototype = apply(prototype)
            return prototype

        for k in prototype.keys():
            if isinstance(prototype[k], dict):
                self._aggregate(prototype[k])
            else:
                prototype[k] = self.aggregator(prototype[k])
                if apply:
                    prototype[k] = apply(prototype[k])
        return prototype

    def stack(self, item_list: list, apply: Callable = None):
        prototype = self._build_prototype(item_list[0])
        for item in item_list:
            self._insert_data(prototype, item)
        if self.aggregator:
            prototype = self._aggregate(prototype, apply=apply)
        return prototype

    def __call__(self, item_list: list, apply: Callable = None):
        return self.stack(item_list, apply=apply)


class OneDepthStacker(Stacker):
    def __init__(self, aggregator: Callable = None):
        super().__init__(aggregator)

    def _build_prototype(self, item):
        if not isinstance(item, dict):
            return []

        prototype = OrderedDict()
        for k in item.keys():
            prototype[k] = []
        return prototype


class FastStacker(Stacker):
    def __init__(self, aggregator: Callable = None):
        super().__init__(aggregator)
        self.prototype = None

    def stack(self, item_list: list, apply: Callable = None):
        if not self.prototype:
            self.prototype = self._build_prototype(item_list[0])
        prototype = copy.deepcopy(self.prototype)
        for item in item_list:
            self._insert_data(prototype, item)
        if self.aggregator:
            prototype = self._aggregate(prototype, apply=apply)
        return prototype


if __name__ == '__main__':

    stacker = Stacker(torch.stack)

    a = torch.tensor([1,2,3])
    b = torch.tensor([1,2,3])
    print(stacker.stack([a]))
