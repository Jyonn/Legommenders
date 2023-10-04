import json
from typing import Dict, Type

import torch
from oba import Obj
from torch import nn

from loader.depot.depot_cache import DepotCache
from loader.depot.fc_unidep import FCUniDep
from loader.global_setting import Setting
from model.common.user_plugin import UserPlugin
from model.inputer.concat_inputer import ConcatInputer
from model.inputer.natural_concat_inputer import NaturalConcatInputer
from model.recommenders.base_recommender import BaseRecommender, BaseRecommenderConfig, RecommenderMeta
from model.utils.column_map import ColumnMap
from loader.embedding.embedding_manager import EmbeddingManager
from model.utils.manager import Manager
from model.utils.nr_dataloader import NRDataLoader
from model.utils.nr_depot import NRDepot
# from loader.recommenders import Recommenders
from loader.base_dataset import BaseDataset
from utils.auto_import import ClassSet
from utils.printer import printer, Color


class DatasetType:
    news = 'news'
    book = 'book'


class Phases:
    train = 'train'
    dev = 'dev'
    test = 'test'
    fast_eval = 'fast_eval'


class Depots:
    def __init__(self, user_data, modes: set, column_map: ColumnMap):
        self.column_map = column_map

        self.train_depot = self.dev_depot = self.test_depot = None
        if Phases.train in modes:
            self.train_depot = DepotCache.get(user_data.depots.train.path, filter_cache=user_data.filter_cache)
        if Phases.dev in modes:
            self.dev_depot = DepotCache.get(user_data.depots.dev.path, filter_cache=user_data.filter_cache)
        if Phases.test in modes:
            self.test_depot = DepotCache.get(user_data.depots.test.path, filter_cache=user_data.filter_cache)

        self.fast_eval_depot = self.create_fast_eval_depot(user_data.depots.dev.path, column_map=column_map)

        self.print = printer[(self.__class__.__name__, '|', Color.BLUE)]

        self.depots = {
            Phases.train: self.train_depot,
            Phases.dev: self.dev_depot,
            Phases.test: self.test_depot,
            Phases.fast_eval: self.fast_eval_depot,
        }  # type: Dict[str, FCUniDep]

        if user_data.union:
            for depot in self.depots.values():
                if not depot:
                    continue
                depot.union(*[DepotCache.get(d) for d in user_data.union])

        if user_data.allowed:
            allowed_list = json.load(open(user_data.allowed))
            for phase in self.depots:
                depot = self.depots[phase]
                if not depot:
                    continue
                sample_num = len(depot)
                super(FCUniDep, depot).filter(lambda x: x in allowed_list, col=depot.id_col)
                self.print(f'Filter {phase} phase with allowed list, sample num: {sample_num} -> {len(depot)}')

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
                        self.print(f'Filter {col} with {filter_str} in {phase} phase, sample num: {sample_num} -> {len(depot)}')

    @staticmethod
    def create_fast_eval_depot(path, column_map: ColumnMap):
        user_depot = FCUniDep(path)
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
        if Setting.simple_dev:
            phases.append(Phases.dev)

        for phase in phases:
            depot = self.depots[phase]
            if not depot:
                continue

            sample_num = len(depot)
            depot.filter('lambda x: x == 1', col=col)
            self.print(f'Filter {col} with x==1 in {phase} phase, sample num: {sample_num} -> {len(depot)}')

    def __getitem__(self, item):
        return self.depots[item]

    def a_depot(self):
        return self.train_depot or self.dev_depot or self.test_depot


class NRDepots:
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

        self.train_nrd = self.dev_nrd = self.test_nrd = None
        if depots.train_depot:
            self.train_nrd = NRDepot(depot=depots.train_depot, order=order, append=append)
        if depots.dev_depot:
            self.dev_nrd = NRDepot(depot=depots.dev_depot, order=order, append=append)
        if depots.test_depot:
            self.test_nrd = NRDepot(depot=depots.test_depot, order=order, append=append)
        self.fast_eval_nrd = NRDepot(depot=depots.fast_eval_depot, order=order, append=append)

        self.nrds = {
            Phases.train: self.train_nrd,
            Phases.dev: self.dev_nrd,
            Phases.test: self.test_nrd,
            Phases.fast_eval: self.fast_eval_nrd,
        }

    def __getitem__(self, item):
        return self.nrds[item]

    def a_nrd(self):
        return self.train_nrd or self.dev_nrd or self.test_nrd


class Datasets:
    def __init__(self, nrds: NRDepots, manager: Manager):
        self.nrds = nrds

        self.train_set = self.dev_set = self.test_set = None
        if nrds.train_nrd:
            self.train_set = BaseDataset(nrd=self.nrds.train_nrd, manager=manager)
        if nrds.dev_nrd:
            self.dev_set = BaseDataset(nrd=self.nrds.dev_nrd, manager=manager)
        if nrds.test_nrd:
            self.test_set = BaseDataset(nrd=self.nrds.test_nrd, manager=manager)

        self.sets = {
            Phases.train: self.train_set,
            Phases.dev: self.dev_set,
            Phases.test: self.test_set,
        }

    def __getitem__(self, item):
        return self.sets[item]

    def a_set(self):
        return self.train_set or self.dev_set or self.test_set


class ConfigManager:
    def __init__(self, data, embed, model, exp):
        self.data = data
        self.embed = embed
        self.model = model
        self.exp = exp
        self.modes = self.parse_mode()

        self.print = printer[(self.__class__.__name__, '|', Color.CYAN)]

        if 'MIND' in self.data.name.upper():
            Setting.dataset = DatasetType.news
        else:
            Setting.dataset = DatasetType.book
        self.print('dataset type: ', Setting.dataset)

        self.print('build column map ...')
        self.column_map = ColumnMap(**Obj.raw(self.data.user))

        self.print('build news and user depots ...')
        self.depots = Depots(user_data=self.data.user, modes=self.modes, column_map=self.column_map)
        self.nrds = NRDepots(depots=self.depots)
        self.doc_nrd = NRDepot(
            depot=self.data.news.depot,
            order=self.data.news.order,
            append=self.data.news.append,
        )
        self.print('doc nrd size: ', len(self.doc_nrd.depot))
        if self.data.news.union:
            for depot in self.data.news.union:
                self.doc_nrd.depot.union(DepotCache.get(depot))

        # for example, PLMNR-NRMS.NRL is a variant of PLMNRNRMS
        self.model_name = self.model.name.split('.')[0].replace('-', '')
        # self.recommender_class = Recommenders()(self.model_name)  # type: Type[BaseRecommender]

        # recommender_set = ClassSet.recommenders()
        operator_set = ClassSet.operators()
        predictor_set = ClassSet.predictors()

        self.item_operator_class = None
        if self.model.meta.item:
            self.item_operator_class = operator_set(self.model.meta.item)
        self.user_operator_class = operator_set(self.model.meta.user)
        self.predictor_class = predictor_set(self.model.meta.predictor)
        self.recommender_meta = RecommenderMeta(
            item_encoder_class=self.item_operator_class,
            user_encoder_class=self.user_operator_class,
            predictor_class=self.predictor_class,
        )

        # self.recommender_class = ClassSet.recommenders()(self.model_name)  # type: Type[BaseRecommender]
        # self.print(f'selected recommender: {str(self.recommender_class.__name__)}')
        self.print(f'Selected Item Encoder: {str(self.item_operator_class.__name__) if self.item_operator_class else "null"}')
        self.print(f'Selected User Encoder: {str(self.user_operator_class.__name__)}')
        self.print(f'Selected Predictor: {str(self.predictor_class.__name__)}')
        self.print(f'Use Negative Sampling: {self.model.config.use_neg_sampling}')
        self.print(f'Use Item Content: {self.model.config.use_news_content}')

        # self.recommender_config = self.recommender_class.config_class(
        #     **Obj.raw(self.model.config),
        # )  # type: BaseRecommenderConfig
        self.recommender_config = BaseRecommenderConfig(**Obj.raw(self.model.config))

        self.print('build embedding manager ...')
        skip_cols = [self.column_map.candidate_col] if self.recommender_config.use_news_content else []
        self.embedding_manager = EmbeddingManager(
            hidden_size=self.recommender_config.embed_hidden_size,
            same_dim_transform=self.model.config.same_dim_transform,
        )

        self.print('load pretrained embeddings ...')
        for embedding_info in self.embed.embeddings:
            self.embedding_manager.load_pretrained_embedding(**Obj.raw(embedding_info))

        self.print('register embeddings ...')
        self.embedding_manager.register_depot(self.nrds.a_nrd(), skip_cols=skip_cols)
        self.embedding_manager.register_vocab(ConcatInputer.vocab)
        if self.model.config.use_news_content:
            self.embedding_manager.register_depot(self.doc_nrd)
            self.embedding_manager.clone_vocab(
                col_name=NaturalConcatInputer.special_col,
                clone_col_name=self.data.news.lm_col or 'title'
            )

        self.print('set <pad> embedding to zeros ...')
        cat_embeddings = self.embedding_manager(ConcatInputer.vocab.name)  # type: nn.Embedding
        cat_embeddings.weight.data[ConcatInputer.PAD] = torch.zeros_like(cat_embeddings.weight.data[ConcatInputer.PAD])

        user_plugin = None
        if self.data.user.plugin:
            self.print(f'user plugin ...')
            user_plugin = UserPlugin(
                depot=DepotCache.get(self.data.user.plugin),
                hidden_size=self.model.config.hidden_size,
                select_cols=self.data.user.plugin_cols,
            )

        self.print('build recommender model and manager ...')
        # self.recommender = self.recommender_class(
        self.recommender = BaseRecommender(
            meta=self.recommender_meta,
            config=self.recommender_config,
            column_map=self.column_map,
            embedding_manager=self.embedding_manager,
            user_nrd=self.nrds.a_nrd(),
            news_nrd=self.doc_nrd,
            user_plugin=user_plugin,
        )
        self.manager = Manager(
            recommender=self.recommender,
            doc_nrd=self.doc_nrd,
            user_nrd=self.nrds.fast_eval_nrd,
        )

        if self.recommender_config.use_neg_sampling:
            self.print('neg sample filtering ...')
            self.depots.negative_filter(self.column_map.label_col)

        if self.exp.policy.use_cache:
            self.print('caching depots ...')
            for depot in self.depots.depots.values():
                depot.start_caching()

        self.print('build datasets ...')
        self.sets = Datasets(nrds=self.nrds, manager=self.manager)

    def parse_mode(self):
        modes = set(self.exp.mode.lower().split('_'))
        if Phases.train in modes:
            modes.add(Phases.dev)
        return modes

    def get_loader(self, phase):
        return NRDataLoader(
            manager=self.manager,
            dataset=self.sets[phase],
            shuffle=phase == Phases.train,
            batch_size=self.exp.policy.batch_size,
            pin_memory=self.exp.policy.pin_memory,
            num_workers=5,
            # collate_fn=self.stacker,
        )
