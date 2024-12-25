import abc
import os

import refconfig
from oba import Obj
from refconfig import RefConfig

from utils.function import argparse
from utils.rand import Rand
from utils.timing import Timing


class CommandInit:
    def __init__(self, required_args, default_args=None, makedirs=None):
        self.required_args = required_args
        self.default_args = default_args or {}
        self.makedirs = makedirs or []

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

        config = Obj(config)

        for makedir in self.makedirs:
            dir_name = config[makedir]
            os.makedirs(dir_name, exist_ok=True)

        return config


class ConfigInit(abc.ABC):
    _d: dict = None

    @classmethod
    def parse(cls):

        if cls._d is not None:
            return cls._d

        cls._d = dict()

        with open(f'.{cls.classname()}') as f:
            config = f.read()

        for line in config.strip().split('\n'):
            key, value = line.split('=')
            cls._d[key.strip().lower()] = value.strip()

        return cls._d

    @classmethod
    def get(cls, key, **kwargs):
        d = cls.parse()

        key = key.lower()
        if key not in d:
            if 'default' in kwargs:
                return kwargs['default']
            raise ValueError(f'key {key} not found in config')

        return d[key]

    @classmethod
    def classname(cls):
        return cls.__name__.lower().replace('init', '')


class DataInit(ConfigInit):
    pass
