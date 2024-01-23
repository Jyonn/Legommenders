import json
from typing import Dict

from pigmento import pnt

from loader.column_map import ColumnMap
from loader.depot.caching_depot import CachingDep
from loader.depot.depot_hub import DepotHub
from loader.meta import Phases, Meta


class Depots:
    def __init__(self, user_data, modes: set, column_map: ColumnMap):
        self.column_map = column_map

        self.train_depot = self.dev_depot = self.test_depot = None
        if Phases.train in modes:
            self.train_depot = DepotHub.get(user_data.depots.train.path, filter_cache=user_data.filter_cache)
        if Phases.dev in modes:
            self.dev_depot = DepotHub.get(user_data.depots.dev.path, filter_cache=user_data.filter_cache)
        if Phases.test in modes:
            self.test_depot = DepotHub.get(user_data.depots.test.path, filter_cache=user_data.filter_cache)

        self.fast_eval_depot = self.create_fast_eval_depot(user_data.depots.dev.path, column_map=column_map)

        self.depots = {
            Phases.train: self.train_depot,
            Phases.dev: self.dev_depot,
            Phases.test: self.test_depot,
            Phases.fast_eval: self.fast_eval_depot,
        }  # type: Dict[str, CachingDep]

        if user_data.union:
            for depot in self.depots.values():
                if not depot:
                    continue
                depot.union(*[DepotHub.get(d) for d in user_data.union])

        if user_data.allowed:
            allowed_list = json.load(open(user_data.allowed))
            for phase in self.depots:
                depot = self.depots[phase]
                if not depot:
                    continue
                sample_num = len(depot)
                super(CachingDep, depot).filter(lambda x: x in allowed_list, col=depot.id_col)
                pnt(f'Filter {phase} phase with allowed list, sample num: {sample_num} -> {len(depot)}')

        if user_data.filters:
            for col in user_data.filters:
                for filter_str in user_data.filters[col]:
                    filter_func_str = f'lambda x: {filter_str}'
                    for phase in [Phases.train, Phases.dev, Phases.test]:
                        depot = self.depots[phase]
                        if not depot:
                            continue
                        sample_num = len(depot)
                        depot.filter(filter_func_str, col=col)
                        pnt(f'Filter {col} with {filter_str} in {phase} phase, sample num: {sample_num} -> {len(depot)}')

        for phase in [Phases.train, Phases.dev, Phases.test]:
            filters = user_data.depots[phase].filters
            depot = self.depots[phase]
            if not depot:
                continue
            for col in filters:
                for filter_str in filters[col]:
                    filter_func_str = f'lambda x: {filter_str}'
                    depot.filter(filter_func_str, col=col)
                    pnt(f'Filter {col} with {filter_str} in {phase} phase, sample num: {len(depot)}')

    @staticmethod
    def create_fast_eval_depot(path, column_map: ColumnMap):
        user_depot = CachingDep(path)
        user_num = user_depot.cols[column_map.user_col].voc.size
        user_depot.reset({
            user_depot.id_col: list(range(user_num)),
            column_map.candidate_col: [[0] for _ in range(user_num)],
            column_map.label_col: [[0] for _ in range(user_num)],
            column_map.user_col: list(range(user_num)),
            column_map.group_col: list(range(user_num)),
        })
        return user_depot

    def negative_filter(self, col):
        phases = [Phases.train]
        if Meta.simple_dev:
            phases.append(Phases.dev)

        for phase in phases:
            depot = self.depots[phase]
            if not depot:
                continue

            sample_num = len(depot)
            depot.filter('lambda x: x == 1', col=col)
            pnt(f'Filter {col} with x==1 in {phase} phase, sample num: {sample_num} -> {len(depot)}')

    def __getitem__(self, item):
        return self.depots[item]

    def a_depot(self):
        return self.train_depot or self.dev_depot or self.test_depot
