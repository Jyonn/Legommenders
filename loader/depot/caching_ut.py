import os.path
import pickle
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


class CachingUT(UniTok):
    """
    A UniDep with filter cache. The filter cache is used to speed up the filtering process.
    """

    filter_cache: bool
    global_filters: list
    job_filters: dict
    filters_base_path: str
    cached_filters_path: str
    cached_filters: list

    @classmethod
    def load(cls, save_dir: str, filter_cache: bool = False):
        ut = super().load(save_dir)
        ut.filter_cache = filter_cache

        # current status
        ut.global_filters = list()
        ut.job_filters = dict()

        # cached status
        ut.filters_base_path = os.path.join(save_dir, 'filters')
        os.makedirs(ut.filters_base_path, exist_ok=True)

        ut.cached_filters_path = os.path.join(ut.filters_base_path, 'filter_cache.pkl')
        ut.cached_filters = []
        ut.load_cache()
        # ut.attempt_update()

        return ut

    # def attempt_update(self):
    #     flag = False
    #     for cached_filter in self.cached_filters:
    #         if cached_filter['path'].endswith('.json'):
    #             json_data = json.load(open(os.path.join(self.filters_base_path, cached_filter['path'])))
    #             numpy_data = np.array(json_data)
    #             np.save(os.path.join(
    #                 self.filters_base_path, cached_filter['path'].replace('.json', '.npy')), numpy_data)
    #             os.remove(os.path.join(self.filters_base_path, cached_filter['path']))
    #             cached_filter['path'] = cached_filter['path'].replace('.json', '.npy')
    #             pnt(f'update filter cache {cached_filter["path"]} on {str(self)} to npy format')
    #             flag = True
    #     if flag:
    #         json.dump(self.cached_filters, cast(SupportsWriteStr, open(self.cached_filters_path, 'w')))

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

        if len(other_col) != len(self.job_filters):
            return False
        for job_name, func_list in other_col.items():
            if job_name not in self.job_filters:
                return False
            if len(func_list) != len(self.job_filters[job_name]):
                return False
            for func in func_list:
                if func not in self.job_filters[job_name]:
                    return False

        return True

    def store_cache(self):
        pnt(f'store filter cache on {str(self)}')

        filter_name = f'{Rand()[6]}.pkl'
        filter_path = os.path.join(self.filters_base_path, filter_name)
        PickleHandler.save(self._legal_indices, filter_path)
        # json.dump(self._legal_indices, cast(SupportsWrite, open(filter_path, 'w')))
        self.cached_filters.append({
            'global': self.global_filters,
            'col': self.job_filters,
            'path': filter_name
        })
        # json.dump(self.cached_filters, cast(SupportsWriteStr, open(self.cached_filters_path, 'w')))
        PickleHandler.save(self.cached_filters, self.cached_filters_path)
        self.load_cache()

    def load_cache(self):
        self.cached_filters = []
        if not self.filter_cache:
            return
        if os.path.exists(self.cached_filters_path):
            # self.cached_filters = json.load(open(self.cached_filters_path))
            self.cached_filters = PickleHandler.load(self.cached_filters_path)
        pnt(f'load {len(self.cached_filters)} filter caches on {str(self)}')

    def filter(self, filter_func: str, col=None):
        if self.filter_cache:
            if col is None:
                if filter_func not in self.global_filters:
                    self.global_filters.append(filter_func)
            else:
                if col not in self.job_filters:
                    self.job_filters[col] = list()
                if filter_func not in self.job_filters[col]:
                    self.job_filters[col].append(filter_func)

            for cached_filter in self.cached_filters:
                if self.is_same_filter(cached_filter):
                    # self._legal_indices = list(np.load(os.path.join(self.filters_base_path, cached_filter['path'])))
                    path = os.path.join(self.filters_base_path, cached_filter['path'])
                    self._legal_indices = PickleHandler.load(path)
                    self._legal_flags = [False] * self._sample_size
                    for index in self._legal_indices:
                        self._legal_flags[index] = True
                    return self

        super().filter(eval(filter_func), col)

        if self.filter_cache:
            self.store_cache()

        return self

    def reset(self, data: dict):
        self.data = data
        self.init_indices()
