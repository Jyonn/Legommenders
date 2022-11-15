import json
from typing import Union

import torch


class Structure:
    @classmethod
    def analyse(cls, x):
        if isinstance(x, dict):
            return cls.analyse_dict(x)
        elif isinstance(x, list) or isinstance(x, tuple) or isinstance(x, set):
            return cls.analyse_list(x)
        elif isinstance(x, torch.Tensor):
            return f'tensor({list(x.shape)})'
        else:
            return type(x).__name__

    @classmethod
    def analyse_dict(cls, d: dict):
        structure = dict()
        for k in d:
            structure[k] = cls.analyse(d[k])
        return structure

    @classmethod
    def analyse_list(cls, l: Union[list, tuple, set]):
        structure = list()
        for x in l:
            structure.append(cls.analyse(x))
        return structure

    @classmethod
    def analyse_and_stringify(cls, x):
        structure = cls.analyse(x)
        return json.dumps(structure, indent=4)
