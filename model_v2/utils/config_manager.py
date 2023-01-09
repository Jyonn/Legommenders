from typing import Dict, Type

from UniTok import UniDep

from loader.base_dataloader import BaseDataLoader
from loader.depot.depot_cache import DepotCache
from model_v2.inputer.concat_inputer import ConcatInputer
from model_v2.recommenders.base_model import BaseRecommender, BaseRecommenderConfig
from model_v2.utils.column_map import ColumnMap
from model_v2.utils.embedding_manager import EmbeddingManager
from model_v2.utils.manager import Manager
from model_v2.utils.nr_dataloader import NRDataLoader
from model_v2.utils.nr_depot import NRDepot
from model_v2.utils.recommenders import Recommenders
from set.base_dataset import BaseDataset


class Phases:
    train = 'train'
    dev = 'dev'
    test = 'test'


class Depots:
    def __init__(self, user_data):
        self.train_depot = DepotCache.get(user_data.depots.train.path)
        self.dev_depot = DepotCache.get(user_data.depots.dev.path)
        self.test_depot = DepotCache.get(user_data.depots.test.path)

        self.depots = {
            Phases.train: self.train_depot,
            Phases.dev: self.dev_depot,
            Phases.test: self.test_depot,
        }  # type: Dict[str, UniDep]

        if user_data.filters:
            for col in user_data.filters:
                for filter_str in user_data.filters[col]:
                    filter_func = eval(f'lambda x: {filter_str}')
                    for depot in self.depots.values():
                        depot.filter(filter_func, col=col)

        if user_data.union:
            for depot in self.depots.values():
                depot.union(*[DepotCache.get(d) for d in user_data.union])

    def negative_filter(self, col):
        for phase in [Phases.train, Phases.dev]:
            self.depots[phase].filter(lambda x: x == 1, col=col)

    def __getitem__(self, item):
        return self.depots[item]


class NRDepots:
    def __init__(self, depots: Depots, column_map: ColumnMap):
        order = [column_map.clicks_col]
        append = [column_map.candidate_col, column_map.label_col]
        self.train_nrd = NRDepot(depot=depots.train_depot, order=order, append=append)
        self.dev_nrd = NRDepot(depot=depots.dev_depot, order=order, append=append)
        self.test_nrd = NRDepot(depot=depots.test_depot, order=order, append=append)

        self.nrds = {
            Phases.train: self.train_nrd,
            Phases.dev: self.dev_nrd,
            Phases.test: self.test_nrd,
        }

    def __getitem__(self, item):
        return self.nrds[item]


class Datasets:
    def __init__(self, nrds: NRDepots, manager: Manager):
        self.nrds = nrds

        self.train_set = BaseDataset(nrd=self.nrds.train_nrd, manager=manager)
        self.dev_set = BaseDataset(nrd=self.nrds.dev_nrd, manager=manager)
        self.test_set = BaseDataset(nrd=self.nrds.test_nrd, manager=manager)

        self.sets = {
            Phases.train: self.train_set,
            Phases.dev: self.dev_set,
            Phases.test: self.test_set,
        }

    def __getitem__(self, item):
        return self.sets[item]


class ConfigManager:
    def __init__(self, data, model, exp):
        self.data = data
        self.model = model
        self.exp = exp

        self.column_map = ColumnMap(
            clicks_col=self.data.user.clicks_col,
            candidate_col=self.data.user.candidate_col,
            label_col=self.data.user.label_col,
        )

        # build news and user depots
        self.depots = Depots(user_data=self.data.user)
        self.nrds = NRDepots(depots=self.depots, column_map=self.column_map)
        self.doc_nrd = NRDepot(
            depot=self.data.news.depot,
            order=self.data.news.order,
            append=self.data.news.append,
        )

        # build embedding manager
        assert self.model.config.news_encoder.hidden_size == self.model.config.user_encoder.hidden_size
        self.embedding_manager = EmbeddingManager(hidden_size=self.model.config.news_encoder.hidden_size)
        self.embedding_manager.register_depot(self.doc_nrd)
        self.embedding_manager.register_depot(self.nrds.train_nrd)
        self.embedding_manager.build_vocab_embedding(
            vocab_name=ConcatInputer.vocab.name,
            vocab_size=ConcatInputer.vocab.get_size(),
        )

        # build recommender model and manager
        self.recommender_class = Recommenders()(self.model.name)  # type: Type[BaseRecommender]
        self.recommender_config = self.recommender_class.config_class(self.model.config)  # type: BaseRecommenderConfig
        self.recommender = self.recommender_class(
            config=self.recommender_config,
            column_map=self.column_map,
            embedding_manager=self.embedding_manager,
        )
        self.manager = Manager(recommender=self.recommender, doc_nrd=self.doc_nrd)

        if self.recommender_config.user_config.negative_sampling:
            self.depots.negative_filter(self.column_map.clicks_col)

        self.sets = Datasets(nrds=self.nrds, manager=self.manager)

    def get_loader(self, phase):
        shuffle = phase != Phases.test
        return NRDataLoader(
            manager=self.manager,
            dataset=self.sets[phase],
            shuffle=shuffle,
            batch_size=self.exp.policy.batch_size,
            pin_memory=self.exp.policy.pin_memory,
        )
