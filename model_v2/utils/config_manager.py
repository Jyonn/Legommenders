from typing import Dict, Type

from UniTok import UniDep
from oba import Obj

from loader.depot.depot_cache import DepotCache
from model_v2.inputer.cat_inputer import CatInputer
from model_v2.recommenders.base_model import BaseRecommender, BaseRecommenderConfig
from model_v2.utils.column_map import ColumnMap
from model_v2.utils.embedding_manager import EmbeddingManager
from model_v2.utils.manager import Manager
from model_v2.utils.nr_dataloader import NRDataLoader
from model_v2.utils.nr_depot import NRDepot
from model_v2.utils.recommenders import Recommenders
from set.base_dataset import BaseDataset
from utils.printer import printer, Color


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

        if user_data.union:
            for depot in self.depots.values():
                depot.union(*[DepotCache.get(d) for d in user_data.union])

        if user_data.filters:
            for col in user_data.filters:
                for filter_str in user_data.filters[col]:
                    filter_func = eval(f'lambda x: {filter_str}')
                    for depot in self.depots.values():
                        depot.filter(filter_func, col=col)

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

        self.print = printer[(self.__class__.__name__, '|', Color.CYAN)]

        self.print('build column map ...')
        self.column_map = ColumnMap(
            clicks_col=self.data.user.clicks_col,
            candidate_col=self.data.user.candidate_col,
            label_col=self.data.user.label_col,
        )

        self.print('build news and user depots ...')
        self.depots = Depots(user_data=self.data.user)
        self.nrds = NRDepots(depots=self.depots, column_map=self.column_map)
        self.doc_nrd = NRDepot(
            depot=self.data.news.depot,
            order=self.data.news.order,
            append=self.data.news.append,
        )

        self.recommender_class = Recommenders()(self.model.name)  # type: Type[BaseRecommender]
        self.print('selected recommender: ', self.recommender_class)
        self.recommender_config = self.recommender_class.config_class(
            news_config=self.model.config.news_encoder,
            user_config=self.model.config.user_encoder,
            use_news_content=self.model.config.use_news_content,
        )  # type: BaseRecommenderConfig

        self.print('build embedding manager ...')
        assert self.model.config.news_encoder.hidden_size == self.model.config.user_encoder.hidden_size
        skip_cols = [self.column_map.candidate_col] if self.recommender_config.use_news_content else []
        self.embedding_manager = EmbeddingManager(hidden_size=self.model.config.news_encoder.hidden_size)
        self.embedding_manager.register_depot(self.doc_nrd)
        self.embedding_manager.register_depot(self.nrds.train_nrd, skip_cols=skip_cols)
        self.embedding_manager.build_vocab_embedding(
            vocab_name=CatInputer.vocab.name,
            vocab_size=CatInputer.vocab.get_size(),
        )

        self.print('build recommender model and manager ...')
        self.recommender = self.recommender_class(
            config=self.recommender_config,
            column_map=self.column_map,
            embedding_manager=self.embedding_manager,
            user_nrd=self.nrds.train_nrd,
            news_nrd=self.doc_nrd,
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
