import os.path
import pickle
from unitok import Symbol
from typing import Protocol, cast, Union

from pigmento import pnt
from unitok import UniTok

from utils.rand import Rand


class SupportsWriteStr(Protocol):
    def write(self, __s: str) -> object:
        ...


class SupportsWriteBytes(Protocol):
    def write(self, __s: bytes) -> object:
        ...


class PickleHandler:
    @staticmethod
    def load(path: str):
        return pickle.load(open(path, "rb"))

    @staticmethod
    def save(data: Union[dict, list], path: str):
        with open(path, "wb") as f:
            pickle.dump(data, cast(SupportsWriteBytes, f))


class LegoUT(UniTok):
    """
    UniTok for Legommenders, supporting filtering cache, and data resetting
    """

    _use_filter_cache: bool
    _general_filters: list
    _job_specific_filters: dict
    _filter_cache_dir: str
    _filter_cache_meta_path: str
    _caches: list
    _selected_attrs: tuple

    gft = Symbol('general_filters')
    jft = Symbol('job_specific_filters')
    pth = Symbol('path')

    @classmethod
    def load(cls, save_dir: str, use_filter_cache: bool = False):
        ut = super().load(save_dir)
        ut._use_filter_cache = use_filter_cache

        # current status
        ut._general_filters = list()
        ut._job_specific_filters = dict()

        ut._filter_cache_dir = os.path.join(save_dir, 'filters')
        os.makedirs(ut._filter_cache_dir, exist_ok=True)

        ut._filter_cache_meta_path = os.path.join(ut._filter_cache_dir, 'filter_cache.pkl')
        ut._caches = []
        ut._load_cache()

        ut._selected_attrs = set()

        return ut

    def _filter_equals(self, other: dict):
        if self.gft.name not in other or self.jft.name not in other or self.pth.name not in other:
            return False

        general_filters = other[self.gft.name]
        job_specific_filters = other[self.jft.name]

        if len(general_filters) != len(self._general_filters):
            return False

        for func in general_filters:
            if func not in self._general_filters:
                return False

        if len(job_specific_filters) != len(self._job_specific_filters):
            return False
        for job_name, func_list in job_specific_filters.items():
            if job_name not in self._job_specific_filters:
                return False
            if len(func_list) != len(self._job_specific_filters[job_name]):
                return False
            for func in func_list:
                if func not in self._job_specific_filters[job_name]:
                    return False

        return True

    def _store_cache(self):
        pnt(f'store filter cache on {str(self)}')

        filter_name = f'{Rand()[6]}.pkl'
        filter_path = os.path.join(self._filter_cache_dir, filter_name)
        PickleHandler.save(self._legal_indices, filter_path)
        # json.dump(self._legal_indices, cast(SupportsWrite, open(filter_path, 'w')))
        self._caches.append({
            self.gft.name: self._general_filters,
            self.jft.name: self._job_specific_filters,
            self.pth.name: filter_name
        })
        # json.dump(self.cached_filters, cast(SupportsWriteStr, open(self.cached_filters_path, 'w')))
        PickleHandler.save(self._caches, self._filter_cache_meta_path)
        self._load_cache()

    def _load_cache(self):
        self._caches = []
        if not self._use_filter_cache:
            return
        if os.path.exists(self._filter_cache_meta_path):
            # self.cached_filters = json.load(open(self.cached_filters_path))
            self._caches = PickleHandler.load(self._filter_cache_meta_path)
        pnt(f'load {len(self._caches)} filter caches on {str(self)}')

    def filter(self, func: str, col=None):
        if self._use_filter_cache:
            if col is None:
                if func not in self._general_filters:
                    self._general_filters.append(func)
            else:
                if col not in self._job_specific_filters:
                    self._job_specific_filters[col] = list()
                if func not in self._job_specific_filters[col]:
                    self._job_specific_filters[col].append(func)

            for cached_filter in self._caches:
                if self._filter_equals(cached_filter):
                    path = os.path.join(self._filter_cache_dir, cached_filter[self.pth.name])
                    self._legal_indices = PickleHandler.load(path)
                    self._legal_flags = [False] * self._sample_size
                    for index in self._legal_indices:
                        self._legal_flags[index] = True
                    return self

        super().filter(eval(func), col)

        if self._use_filter_cache:
            self._store_cache()

        return self

    def reset(self, data: dict):
        self.data = data
        self.init_indices()

    def rebuild(self, selected_attrs: dict):
        attrs = set()
        for attr in selected_attrs:
            attrs.add(attr)
            truncate = selected_attrs[attr]
            if truncate:
                self.retruncate(attr, truncate)
        self._selected_attrs = tuple(attrs)

    def __getitem__(self, item):
        if len(self._selected_attrs):
            return super().__getitem__((item, self._selected_attrs))
        return super().__getitem__(item)
