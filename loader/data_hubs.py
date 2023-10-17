from loader.data_hub import DataHub
from loader.depots import Depots
from loader.meta import Phases


class DataHubs:
    def __init__(self, depots: Depots):
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
        if depots.train_depot:
            self.train_hub = DataHub(depot=depots.train_depot, order=order, append=append)
        if depots.dev_depot:
            self.dev_hub = DataHub(depot=depots.dev_depot, order=order, append=append)
        if depots.test_depot:
            self.test_hub = DataHub(depot=depots.test_depot, order=order, append=append)
        self.fast_eval_hub = DataHub(depot=depots.fast_eval_depot, order=order, append=append)

        self.hubs = {
            Phases.train: self.train_hub,
            Phases.dev: self.dev_hub,
            Phases.test: self.test_hub,
            Phases.fast_eval: self.fast_eval_hub,
        }

    def __getitem__(self, item):
        return self.hubs[item]

    def a_hub(self):
        return self.train_hub or self.dev_hub or self.test_hub
