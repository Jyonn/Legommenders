import os

from UniTok import UniDep

from loader.global_setting import Setting
from utils.splitter import Splitter


class FilterUniDep(UniDep):
    def __init__(self, store_dir):
        super().__init__(store_dir=store_dir)

    def customize(self, col, lambda_detector):
        valid_sample_indexes = []

        for sample in self:
            if lambda_detector(sample[col]):
                valid_sample_indexes.append(sample[self.id_col])
        self.index_order = valid_sample_indexes
        self.sample_size = len(self.index_order)

    @classmethod
    def from_config(cls, store, sub_folder=None, filters=None):
        data_dir = store.data_dir
        if sub_folder:
            data_dir = os.path.join(data_dir, sub_folder)

        depot = cls(store_dir=data_dir)
        if store.union:
            depot.union(*[UniDep(d) for d in store.union])

        if filters:
            for col in filters:
                for filtering in filters[col]:
                    if filtering == 'remove_empty':
                        filtering = 'x'
                    filtering = eval(f'lambda x: {filtering}')
                    depot.customize(col, filtering)
        return depot

    @classmethod
    def parse(cls, data):
        depots = dict()
        splitter = None

        if data.has_split:
            for mode in data.split:
                filters = data.filter[mode]
                depots[mode] = cls.from_config(
                    store=data.store,
                    sub_folder=data.split[mode].path,
                    filters=filters,
                )
        else:
            filters = data.filter
            depot = cls.from_config(
                store=data.store,
                filters=filters
            )
            splitter = Splitter()
            for mode in data.split:
                assert mode in Setting.MODES

                splitter.add(
                    name=mode,
                    weight=data.split[mode].weight
                )
                depots[mode] = depot

        return depots, splitter
