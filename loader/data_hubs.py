from loader.data_hub import DataHub
from loader.uts import UTs
from loader.meta import LegoSymbols


class DataHubs:
    def __init__(self, depots: UTs):
        column_map = depots.column_map

        order = [column_map.clicks_col]
        append = [
            column_map.candidate_col,
            column_map.label_col,
            column_map.group_col,
            column_map.user_col,
        ]
        if column_map.fake_col:
            append.append(column_map.fake_col)
        if column_map.neg_col:
            append.append(column_map.neg_col)

        self.train_hub = self.dev_hub = self.test_hub = None
        if depots.train_ut:
            self.train_hub = DataHub(ut=depots.train_ut, order=order, append=append)
        if depots.dev_ut:
            self.dev_hub = DataHub(ut=depots.dev_ut, order=order, append=append)
        if depots.test_ut:
            self.test_hub = DataHub(ut=depots.test_ut, order=order, append=append)
        self.fast_eval_hub = DataHub(ut=depots.fast_eval_ut, order=order, append=append)

        self.hubs = {
            LegoSymbols.train: self.train_hub,
            LegoSymbols.dev: self.dev_hub,
            LegoSymbols.test: self.test_hub,
            LegoSymbols.fast_eval: self.fast_eval_hub,
        }

    def __getitem__(self, item):
        return self.hubs[item]

    def a_hub(self):
        return self.train_hub or self.dev_hub or self.test_hub
