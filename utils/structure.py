import json

import torch


class Structure:
    @classmethod
    def analyse(cls, d: dict):
        structure = dict()

        for k in d:
            if isinstance(d[k], dict):
                structure[k] = cls.analyse(d[k])
            elif isinstance(d[k], torch.Tensor):
                structure[k] = f'tensor({list(d[k].shape)})'
            else:
                structure[k] = type(d[k]).__name__
        return structure

    @classmethod
    def analyse_and_stringify(cls, d: dict):
        structure = cls.analyse(d)
        return json.dumps(structure, indent=4)
