import os

import refconfig
from oba import Obj
from refconfig import RefConfig
from smartdict import DictCompiler

from utils.function import argparse
from utils.rand import Rand
from utils.timing import Timing


class PathSearcher(DictCompiler):
    compiler = DictCompiler({})

    @classmethod
    def search(cls, d: dict, path: str):
        cls.compiler.d = d
        cls.compiler.circle = {}
        return cls.compiler._get_value(path)


class ConfigInit:
    def __init__(self, required_args, default_args, makedirs):
        self.required_args = required_args
        self.default_args = default_args
        self.makedirs = makedirs

    def parse(self):
        kwargs = argparse()

        for arg in self.required_args:
            if arg not in kwargs:
                raise ValueError(f'miss argument {arg}')

        for arg in self.default_args:
            if arg not in kwargs:
                kwargs[arg] = self.default_args[arg]

        config = RefConfig().add(refconfig.CType.SMART, **kwargs)
        config = config.add(refconfig.CType.RAW, rand=Rand(), time=Timing()).parse()

        for makedir in self.makedirs:
            dir_name = PathSearcher.search(config, makedir)
            os.makedirs(dir_name, exist_ok=True)

        config = Obj(config)
        return config
        # data, model, exp = config.data, config.model, config.exp
        # return config, data, model, exp
