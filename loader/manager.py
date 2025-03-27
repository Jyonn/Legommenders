from typing import Type, Optional

from pigmento import pnt
from torch.utils.data import DataLoader
from unitok import Symbol

from loader.class_hub import ClassHub
from loader.column_map import ColumnMap
from loader.data_set import DataSet
from loader.embedding_hub import EmbeddingHub
from loader.env import Env
from loader.resampler import Resampler
from loader.symbols import Symbols
from loader.ut.lego_ut import LegoUT
from loader.ut.ut_hub import UTHub
from model.lego_config import LegoConfig

from model.legommender import Legommender
from model.operators.base_operator import BaseOperator
from model.predictors.base_predictor import BasePredictor
from utils.structure import Structure


class Manager:
    item_ut: LegoUT
    user_ut: LegoUT
    fast_ut: LegoUT  # for fast evaluation
    item_inputs: list[str]
    inter_uts: dict[str, LegoUT]
    inter_sets: dict[str, DataSet]
    cm: ColumnMap

    item_operator_class: Optional[Type[BaseOperator]]
    user_operator_class: Type[BaseOperator]
    predictor_class: Type[BasePredictor]
    lego_config: LegoConfig

    modes = {Symbols.train, Symbols.dev, Symbols.test}

    embedding_hub: EmbeddingHub

    checkpoint_paths: list[str]

    def __init__(self, data, embed, model, exp):
        self.data = data
        self.embed = embed
        self.model = model
        self.exp = exp

        self.load_data()
        self.load_model_configs()
        self.load_embeddings()
        self.lego_config.build_components()
        self.lego_config.register_inputer_vocabs()

        self.legommender = Legommender(self.lego_config)
        self.resampler = Resampler(self.lego_config)

        self.load_datasets()
        # self.load_checkpoint_paths()

    def load_items(self):
        ut = UTHub.get(self.data.item.ut)
        inputs = self.data.item.inputs

        input_cols = []
        selected_attrs = dict()
        for value in inputs:
            if isinstance(value, dict):
                if len(value) != 1:
                    raise ValueError(f'Invalid input column: {inputs}')
                col, truncate = list(value.items())[0]
            else:
                col, truncate = value, None

            with ut:
                current_col = None
                if '->' in col:
                    current_col, col = list(map(str.strip, col.split('->')))
                    ut.replicate(current_col, col, lazy=False)
                elif '-->' in col:
                    current_col, col = list(map(str.strip, col.split('-->')))
                    ut.replicate(current_col, col, lazy=True)
                if current_col is not None:
                    pnt(f'Replicated {current_col} to {col}')

            selected_attrs[col] = truncate
            input_cols.append(col)

        selected_attrs[ut.key_job.name] = None
        ut.rebuild(selected_attrs)
        return ut, input_cols

    def load_users(self):
        ut = UTHub.get(self.data.user.ut)
        if self.data.user.truncate:
            ut.retruncate(
                job=ut.meta.jobs[self.cm.history_col],
                truncate=self.data.user.truncate
            )
        return ut

    def load_interactions(self):
        uts = dict()
        for mode in self.modes:
            uts[mode] = UTHub.get(self.data.inter[mode.name], use_filter_cache=True)
        return uts

    def create_fast_ut(self):
        ut: LegoUT = LegoUT.load(self.data.inter.test)
        user_num = len(self.user_ut)
        reset_data = {
            ut.key_job.name: list(range(user_num)),
            self.cm.item_col: [[0] for _ in range(user_num)],
            self.cm.user_col: list(range(user_num)),
            self.cm.label_col: [[0] for _ in range(user_num)],
        }
        # the group column may be the same as the user column
        reset_data.update({
            self.cm.group_col: list(range(user_num)),
        })
        ut.reset(reset_data)
        return ut

    def load_data(self):
        self.cm = ColumnMap(**self.data.column_map())

        self.item_ut, self.item_inputs = self.load_items()
        self.user_ut = self.load_users()
        self.inter_uts = self.load_interactions()
        self.fast_ut = self.create_fast_ut()
        self.inter_uts[Symbols.fast_eval] = self.fast_ut

        for mode in self.inter_uts:
            with self.inter_uts[mode] as ut:
                ut.union(self.user_ut, soft_union=False)

        if self.data.inter.filters:
            for col in self.data.inter.filters:
                for filter_str in self.data.inter.filters[col]:
                    for mode in self.inter_uts:
                        ut = self.inter_uts[mode]
                        sample_num = len(ut)
                        ut.filter(filter_str, col=col)
                        pnt(f'Filter {col} with {filter_str} in {mode} phase, sample num: {sample_num} -> {len(ut)}')

        self.cm.set_column_vocab(self.fast_ut)

    def load_model_configs(self):
        operators = ClassHub.operators()
        predictors = ClassHub.predictors()

        self.item_operator_class = None
        if self.model.meta.item:
            self.item_operator_class = operators(self.model.meta.item)
        self.user_operator_class = operators(self.model.meta.user)
        self.predictor_class = predictors(self.model.meta.predictor)

        pnt(f'Selected Item Encoder: {str(self.item_operator_class.__name__) if self.item_operator_class else "null"}')
        pnt(f'Selected User Encoder: {str(self.user_operator_class.__name__)}')
        pnt(f'Selected Predictor: {str(self.predictor_class.__name__)}')

        self.lego_config = LegoConfig(**self.model.config())
        self.lego_config.set_component_classes(
            item_operator_class=self.item_operator_class,
            user_operator_class=self.user_operator_class,
            predictor_class=self.predictor_class,
        )
        self.lego_config.set_item_ut(self.item_ut, self.item_inputs)
        self.lego_config.set_user_ut(self.user_ut, [self.cm.history_col])
        self.lego_config.set_column_map(self.cm)

        if self.lego_config.use_neg_sampling:
            self.negative_sampling()

    def load_embeddings(self):
        self.embedding_hub = EmbeddingHub(
            embedding_dim=self.lego_config.item_hidden_size,
            transformation=self.embed.transformation,
            transformation_dropout=self.embed.transformation_dropout,
        )
        for info in self.embed.embeddings:
            self.embedding_hub.load_pretrained_embedding(**info())
        if self.lego_config.use_item_content:
            self.embedding_hub.register_ut(self.item_ut, self.item_inputs)
        else:
            self.embedding_hub.register_vocab(self.item_ut.key_job.tokenizer.vocab)

        self.lego_config.set_embedding_hub(self.embedding_hub)

    def negative_sampling(self):
        modes = [Symbols.train]
        if self.exp.policy.simple_dev:
            modes.append(Symbols.dev)

        filter_str = 'lambda x: x == 1'

        for mode in modes:
            ut = self.inter_uts[mode]
            if not ut:
                continue
            sample_num = len(ut)
            ut.filter(filter_str, col=self.cm.label_col)
            pnt(f'Filter {self.cm.label_col} with {filter_str} in {mode} phase, sample num: {sample_num} -> {len(ut)}')

    def load_datasets(self):
        self.inter_sets = dict()
        for mode in self.inter_uts:
            ut = self.inter_uts[mode]
            self.inter_sets[mode] = DataSet(ut, resampler=self.resampler)

    def stringify(self):
        for mode in self.modes:
            if mode in self.inter_uts:
                pnt(self.inter_uts[mode][0])
                break

        for mode in self.modes:
            if mode in self.inter_uts:
                pnt(Structure().analyse_and_stringify(self.inter_sets[mode][0]))
                break

    def _get_loader(self, mode: Symbol):
        return DataLoader(
            dataset=self.inter_sets[mode],
            shuffle=mode is Symbols.train,
            batch_size=self.exp.policy.batch_size,
            pin_memory=self.exp.policy.pin_memory,
            num_workers=5,
        )

    @property
    def train_ut(self):
        return self.inter_uts[Symbols.train]

    @property
    def dev_ut(self):
        return self.inter_uts[Symbols.dev]

    @property
    def test_ut(self):
        return self.inter_uts[Symbols.test]

    @property
    def train_set(self):
        return self.inter_sets[Symbols.train]

    @property
    def dev_set(self):
        return self.inter_sets[Symbols.dev]

    @property
    def test_set(self):
        return self.inter_sets[Symbols.test]

    def get_train_loader(self, setup=Symbols.train):
        self.setup(setup)
        return self._get_loader(Symbols.train)

    def get_dev_loader(self, setup=Symbols.dev):
        self.setup(setup)
        return self._get_loader(Symbols.dev)

    def get_test_loader(self, setup=Symbols.test):
        self.setup(setup)
        return self._get_loader(Symbols.test)

    def setup(self, mode):
        if mode is Symbols.train:
            Env.train()
            self.legommender.train().cacher.clean()
            return

        if mode is Symbols.dev:
            Env.dev()
        else:
            Env.test()

        self.legommender.eval().cacher.cache(
            item_contents=self.resampler.item_cache,
            user_contents=self.inter_sets[Symbols.fast_eval],
        )
