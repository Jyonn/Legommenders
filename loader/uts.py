import json
from typing import Dict

from pigmento import pnt

from loader.column_map import ColumnMap
from loader.depot.caching_ut import CachingUT
from loader.depot.ut_hub import UTHub
from loader.meta import Meta, LegoSymbols


class UTs:
    def __init__(self, user_data, modes: set, column_map: ColumnMap):
        self.column_map = column_map

        self.train_ut = self.dev_ut = self.test_ut = None
        if LegoSymbols.train in modes:
            self.train_ut = UTHub.get(user_data.depots.train.path, filter_cache=user_data.filter_cache)
        if LegoSymbols.dev in modes:
            self.dev_ut = UTHub.get(user_data.depots.dev.path, filter_cache=user_data.filter_cache)
        if LegoSymbols.test in modes:
            self.test_ut = UTHub.get(user_data.depots.test.path, filter_cache=user_data.filter_cache)

        self.fast_eval_ut = self.create_fast_eval_depot(user_data.depots.dev.path, column_map=column_map)

        self.uts = {
            LegoSymbols.train.name: self.train_ut,
            LegoSymbols.dev.name: self.dev_ut,
            LegoSymbols.test.name: self.test_ut,
            LegoSymbols.fast_eval.name: self.fast_eval_ut,
        }  # type: Dict[str, CachingUT]

        if user_data.union:
            for ut in self.uts.values():
                if not ut:
                    continue
                for other in user_data.union:
                    other_ut = UTHub.get(other)
                    with ut:
                        ut.union(other_ut, soft_union=False)

        if user_data.allowed:
            allowed_list = json.load(open(user_data.allowed))
            for phase in self.uts:
                ut = self.uts[phase]
                if not ut:
                    continue
                sample_num = len(ut)
                super(CachingUT, ut).filter(lambda x: x in allowed_list, job=ut.key_job)
                pnt(f'Filter {phase} phase with allowed list, sample num: {sample_num} -> {len(ut)}')

        if user_data.filters:
            for col in user_data.filters:
                for filter_str in user_data.filters[col]:
                    filter_func_str = f'lambda x: {filter_str}'
                    for phase in [LegoSymbols.train.name, LegoSymbols.dev.name, LegoSymbols.test.name]:
                        ut = self.uts[phase]
                        if not ut:
                            continue
                        sample_num = len(ut)
                        ut.filter(filter_func_str, col=col)
                        pnt(f'Filter {col} with {filter_str} in {phase} phase, sample num: {sample_num} -> {len(ut)}')

        for phase in [LegoSymbols.train, LegoSymbols.dev, LegoSymbols.test]:
            filters = user_data.uts[phase].filters
            ut = self.uts[phase]
            if not ut:
                continue
            for col in filters:
                for filter_str in filters[col]:
                    filter_func_str = f'lambda x: {filter_str}'
                    ut.filter(filter_func_str, col=col)
                    pnt(f'Filter {col} with {filter_str} in {phase} phase, sample num: {len(ut)}')

    @staticmethod
    def create_fast_eval_depot(path, column_map: ColumnMap):
        user_ut: CachingUT = CachingUT.load(path)
        user_num = user_ut.meta.jobs[column_map.user_col].tokenizer.vocab.size
        user_ut.reset({
            user_ut.key_job.name: list(range(user_num)),
            column_map.candidate_col: [[0] for _ in range(user_num)],
            column_map.label_col: [[0] for _ in range(user_num)],
            column_map.user_col: list(range(user_num)),
            column_map.group_col: list(range(user_num)),
        })
        return user_ut

    def negative_filter(self, col):
        phases = [LegoSymbols.train]
        if Meta.simple_dev:
            phases.append(LegoSymbols.dev)

        for phase in phases:
            depot = self.uts[phase]
            if not depot:
                continue

            sample_num = len(depot)
            depot.filter('lambda x: x == 1', col=col)
            pnt(f'Filter {col} with x==1 in {phase} phase, sample num: {sample_num} -> {len(depot)}')

    def __getitem__(self, item):
        return self.uts[item]

    def a_depot(self):
        return self.train_ut or self.dev_ut or self.test_ut
