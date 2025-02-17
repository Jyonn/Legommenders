import abc
import os.path
import warnings
from typing import cast

import refconfig
from oba import Obj
from refconfig import RefConfig

from utils.function import argparse, package_require

package_require('refconfig', '0.1.0')
package_require('smartdict', '0.2.1')
package_require('unitok', '4.3.6')


class CommandInit:
    def __init__(self, required_args, default_args=None):
        self.required_args = required_args
        self.default_args = default_args or {}

    def parse(self):
        kwargs = argparse()

        for arg in self.required_args:
            if arg not in kwargs:
                raise ValueError(f'miss argument {arg}')

        for arg in self.default_args:
            if arg not in kwargs:
                kwargs[arg] = self.default_args[arg]

        config = RefConfig().add(refconfig.CType.SMART, **kwargs)
        config = config.add(refconfig.CType.RAW).parse()

        config = Obj(config)

        return config


class ConfigInit(abc.ABC):
    _d: dict = None

    @classmethod
    def parse(cls):

        if cls._d is not None:
            return cls._d

        cls._d = dict()

        if not os.path.exists(f'.{cls.classname()}'):
            warnings.warn(f'config file .{cls.classname()} not found')
            print(f'example .{cls.classname()} config:')
            print(cls.examples())
            return cls._d

        with open(f'.{cls.classname()}') as f:
            config = f.read()

        for line in config.strip().split('\n'):  # type: str
            key, value = line.split('=')
            cls._d[key.strip().lower()] = cast(str, value).strip()

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

    @classmethod
    def examples(cls):
        raise NotImplementedError


class DataInit(ConfigInit):
    @classmethod
    def examples(cls):
        return '\n'.join(['mind = /path/to/mind', 'goodreads = /path/to/goodreads'])


class ModelInit(ConfigInit):
    @classmethod
    def examples(cls):
        return '\n'.join(['bertbase = bert-base-uncased', 'bertlarge = bert-large-uncased'])


class AuthInit(ConfigInit):
    @classmethod
    def examples(cls):
        return '\n'.join(['user = user', 'password = password'])
