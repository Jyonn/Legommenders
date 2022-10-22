import os

import refconfig
from oba import Obj
from refconfig import RefConfig
from smartdict import DictCompiler

from utils.dynamic_parser import DynamicParser


# makedirs = [
#     'data.store.save_dir',
# ]


class PathSearcher(DictCompiler):
    compiler = DictCompiler({})

    @classmethod
    def search(cls, d: dict, path: str):
        cls.compiler.d = d
        cls.compiler.circle = {}
        return cls.compiler._get_value(path)


class ConfigInit:
    def __init__(self, required_args, makedirs):
        # required_args = ['data', 'model', 'exp']
        self.required_args = required_args
        self.makedirs = makedirs

    def parse(self):
        kwargs = DynamicParser.parse()

        for arg in self.required_args:
            if arg not in kwargs:
                raise ValueError(f'miss argument {arg}')

        config = RefConfig().add(refconfig.CType.SMART, **kwargs).parse()

        for makedir in self.makedirs:
            dir_name = PathSearcher.search(config, makedir)
            os.makedirs(dir_name, exist_ok=True)

        config = Obj(config)
        return config
        # data, model, exp = config.data, config.model, config.exp
        # return config, data, model, exp
