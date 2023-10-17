import json
import os.path

import numpy as np
from UniTok import UniDep
from pigmento import pnt

from utils.rand import Rand


class CachingDep(UniDep):
    """
    A UniDep with filter cache. The filter cache is used to speed up the filtering process.
    """

    def __init__(self, store_dir, filter_cache: bool = False, **kwargs):
        super().__init__(store_dir, **kwargs)

        self.filter_cache = filter_cache

        # current status
        self.global_filters = list()
        self.col_filters = dict()

        # cached status
        self.filters_base_path = os.path.join(self.store_dir, 'filters')
        os.makedirs(self.filters_base_path, exist_ok=True)

        self.cached_filters_path = os.path.join(self.filters_base_path, 'filter_cache.json')
        self.cached_filters = []
        self.load_cache()
        self.attempt_update()

    def attempt_update(self):
        flag = False
        for cached_filter in self.cached_filters:
            if cached_filter['path'].endswith('.json'):
                json_data = json.load(open(os.path.join(self.filters_base_path, cached_filter['path'])))
                numpy_data = np.array(json_data)
                np.save(os.path.join(
                    self.filters_base_path, cached_filter['path'].replace('.json', '.npy')), numpy_data)
                os.remove(os.path.join(self.filters_base_path, cached_filter['path']))
                cached_filter['path'] = cached_filter['path'].replace('.json', '.npy')
                pnt(f'update filter cache {cached_filter["path"]} on {str(self)} to npy format')
                flag = True
        if flag:
            json.dump(self.cached_filters, open(self.cached_filters_path, 'w'))

    def is_same_filter(self, other: dict):
        other_global, other_col = other['global'], other['col']
        # compare other_global with self.global_filters
        # value order is not important
        if len(other_global) != len(self.global_filters):
            return False

        for func in other_global:
            if func not in self.global_filters:
                return False

        # compare other_col with self.col_filters
        # value order is not important

        if len(other_col) != len(self.col_filters):
            return False
        for col_name, func_list in other_col.items():
            if col_name not in self.col_filters:
                return False
            if len(func_list) != len(self.col_filters[col_name]):
                return False
            for func in func_list:
                if func not in self.col_filters[col_name]:
                    return False

        return True

    def store_cache(self):
        pnt(f'store filter cache on {str(self)}')

        filter_name = f'{Rand()[6]}.json'
        filter_path = os.path.join(self.filters_base_path, filter_name)
        self._indexes = [int(i) for i in self._indexes]
        json.dump(self._indexes, open(filter_path, 'w'))
        self.cached_filters.append({
            'global': self.global_filters,
            'col': self.col_filters,
            'path': filter_name
        })
        json.dump(self.cached_filters, open(self.cached_filters_path, 'w'))
        self.load_cache()

    def load_cache(self):
        self.cached_filters = []
        if not self.filter_cache:
            return
        if os.path.exists(self.cached_filters_path):
            self.cached_filters = json.load(open(self.cached_filters_path))
        pnt(f'load {len(self.cached_filters)} filter caches on {str(self)}')

    def filter(self, filter_func: str, col=None):
        if self.filter_cache:
            if col is None:
                if filter_func not in self.global_filters:
                    self.global_filters.append(filter_func)
            else:
                if col not in self.col_filters:
                    self.col_filters[col] = list()
                if filter_func not in self.col_filters[col]:
                    self.col_filters[col].append(filter_func)

            for cached_filter in self.cached_filters:
                if self.is_same_filter(cached_filter):
                    self._indexes = list(np.load(os.path.join(self.filters_base_path, cached_filter['path'])))
                    self.sample_size = len(self._indexes)
                    return self

        super().filter(eval(filter_func), col)

        if self.filter_cache:
            self.store_cache()

        return self
