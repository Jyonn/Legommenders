from typing import Dict, Type

import torch
from UniTok import UniDep
from oba import Obj
from torch import nn

from loader.depot.depot_cache import DepotCache
from model_v2.inputer.concat_inputer import ConcatInputer
from model_v2.recommenders.base_recommender import BaseRecommender, BaseRecommenderConfig
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
        self.train_depot = DepotCache.get(user_data.depots.train.path, filter_cache=user_data.filter_cache)
        self.dev_depot = DepotCache.get(user_data.depots.dev.path, filter_cache=user_data.filter_cache)
        self.test_depot = DepotCache.get(user_data.depots.test.path, filter_cache=user_data.filter_cache)

        self.print = printer[(self.__class__.__name__, '|', Color.BLUE)]

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
                    filter_func_str = f'lambda x: {filter_str}'
                    for phase in self.depots:
                        depot = self.depots[phase]
                        sample_num = len(depot)
                        depot.filter(filter_func_str, col=col)
                        self.print(f'Filter {col} with {filter_str} in {phase} phase, sample num: {sample_num} -> {len(depot)}')

    def negative_filter(self, col):
        for phase in [Phases.train, Phases.dev]:
            depot = self.depots[phase]
            sample_num = len(depot)
            depot.filter('lambda x: x == 1', col=col)
            self.print(f'Filter {col} with x==1 in {phase} phase, sample num: {sample_num} -> {len(depot)}')

    def __getitem__(self, item):
        return self.depots[item]


class NRDepots:
    def __init__(self, depots: Depots, column_map: ColumnMap):
        order = [column_map.clicks_col]
        append = [
            column_map.candidate_col,
            column_map.label_col,
            column_map.neg_col,
            column_map.group_col,
            column_map.user_col,
        ]
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
    def __init__(self, data, embed, model, exp):
        self.data = data
        self.embed = embed
        self.model = model
        self.exp = exp

        self.print = printer[(self.__class__.__name__, '|', Color.CYAN)]

        self.print('build column map ...')
        self.column_map = ColumnMap(
            clicks_col=self.data.user.clicks_col,
            candidate_col=self.data.user.candidate_col,
            label_col=self.data.user.label_col,
            neg_col=self.data.user.neg_col,
            group_col=self.data.user.group_col,
            user_col=self.data.user.user_col,
        )

        self.print('build news and user depots ...')
        self.depots = Depots(user_data=self.data.user)
        self.nrds = NRDepots(depots=self.depots, column_map=self.column_map)
        self.doc_nrd = NRDepot(
            depot=self.data.news.depot,
            order=self.data.news.order,
            append=self.data.news.append,
        )

        # for example, PLMNR-NRMS.NRL is a variant of PLMNRNRMS
        self.model_name = self.model.name.split('.')[0].replace('-', '')
        self.recommender_class = Recommenders()(self.model_name)  # type: Type[BaseRecommender]
        self.print(f'selected recommender: {self.recommender_class}')
        self.recommender_config = self.recommender_class.config_class(
            **Obj.raw(self.model.config),
        )  # type: BaseRecommenderConfig

        self.print('build embedding manager ...')
        skip_cols = [self.column_map.candidate_col] if self.recommender_config.use_news_content else []
        self.embedding_manager = EmbeddingManager(hidden_size=self.model.config.hidden_size)

        self.print('load pretrained embeddings ...')
        for embedding_info in self.embed.embeddings:
            self.embedding_manager.load_pretrained_embedding(**Obj.raw(embedding_info))

        self.print('register embeddings ...')
        if self.model.config.use_news_content:
            self.embedding_manager.register_depot(self.doc_nrd)
        self.embedding_manager.register_depot(self.nrds.train_nrd, skip_cols=skip_cols)
        self.embedding_manager.register_vocab(ConcatInputer.vocab)

        self.print('set <pad> embedding to zeros ...')
        cat_embeddings = self.embedding_manager(ConcatInputer.vocab.name)  # type: nn.Embedding
        cat_embeddings.weight.data[ConcatInputer.PAD] = torch.zeros_like(cat_embeddings.weight.data[ConcatInputer.PAD])

        self.print('build recommender model and manager ...')
        self.recommender = self.recommender_class(
            config=self.recommender_config,
            column_map=self.column_map,
            embedding_manager=self.embedding_manager,
            user_nrd=self.nrds.train_nrd,
            news_nrd=self.doc_nrd,
        )
        self.manager = Manager(recommender=self.recommender, doc_nrd=self.doc_nrd)

        if self.recommender_class.use_neg_sampling:
            self.print('neg sample filtering ...')
            self.depots.negative_filter(self.column_map.label_col)

        self.print('caching depots ...')
        for depot in self.depots.depots.values():
            depot.start_caching()

        self.print('build datasets ...')
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
