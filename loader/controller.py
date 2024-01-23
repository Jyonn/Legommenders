import warnings

import torch
from oba import Obj
from pigmento import pnt
from torch import nn

from loader.data_hubs import DataHubs
from loader.data_sets import DataSets
from loader.depot.depot_hub import DepotHub
from loader.depots import Depots
from loader.meta import Meta, Phases, DatasetType
from loader.status import Status
from model.common.user_plugin import UserPlugin
from model.inputer.concat_inputer import ConcatInputer
from model.inputer.flatten_seq_inputer import FlattenSeqInputer
from model.inputer.natural_concat_inputer import NaturalConcatInputer
from model.legommender import Legommender, LegommenderConfig, LegommenderMeta
from loader.column_map import ColumnMap
from loader.embedding.embedding_hub import EmbeddingHub
from loader.resampler import Resampler
from loader.data_loader import DataLoader
from loader.data_hub import DataHub
from loader.class_hub import ClassHub


class Controller:
    def __init__(self, data, embed, model, exp):
        self.data = data
        self.embed = embed
        self.model = model
        self.exp = exp
        self.modes = self.parse_mode()

        self.status = Status()

        if 'MIND' in self.data.name.upper():
            Meta.data_type = DatasetType.news
        else:
            Meta.data_type = DatasetType.book
        pnt('dataset type: ', Meta.data_type)

        pnt('build column map ...')
        self.column_map = ColumnMap(**Obj.raw(self.data.user))

        # depots and data hubs initialization
        self.depots = Depots(user_data=self.data.user, modes=self.modes, column_map=self.column_map)
        self.hubs = DataHubs(depots=self.depots)
        self.item_hub = DataHub(
            depot=self.data.item.depot,
            order=self.data.item.order,
            append=self.data.item.append,
        )
        if self.data.item.union:
            for depot in self.data.item.union:
                self.item_hub.depot.union(DepotHub.get(depot))

        # legommender components initialization
        operator_set = ClassHub.operators()
        predictor_set = ClassHub.predictors()

        self.item_operator_class = None
        if self.model.meta.item:
            self.item_operator_class = operator_set(self.model.meta.item)
        self.user_operator_class = operator_set(self.model.meta.user)
        self.predictor_class = predictor_set(self.model.meta.predictor)
        self.legommender_meta = LegommenderMeta(
            item_encoder_class=self.item_operator_class,
            user_encoder_class=self.user_operator_class,
            predictor_class=self.predictor_class,
        )

        pnt(f'Selected Item Encoder: {str(self.item_operator_class.__name__) if self.item_operator_class else "null"}')
        pnt(f'Selected User Encoder: {str(self.user_operator_class.__name__)}')
        pnt(f'Selected Predictor: {str(self.predictor_class.__name__)}')
        pnt(f'Use Negative Sampling: {self.model.config.use_neg_sampling}')
        pnt(f'Use Item Content: {self.model.config.use_item_content}')

        self.legommender_config = LegommenderConfig(**Obj.raw(self.model.config))

        # embedding initialization
        skip_cols = [self.column_map.candidate_col] if self.legommender_config.use_item_content else []
        self.embedding_hub = EmbeddingHub(
            hidden_size=self.legommender_config.embed_hidden_size,
            same_dim_transform=self.model.config.same_dim_transform,
        )
        for embedding_info in self.embed.embeddings:
            self.embedding_hub.load_pretrained_embedding(**Obj.raw(embedding_info))
        self.embedding_hub.register_depot(self.hubs.a_hub(), skip_cols=skip_cols)
        self.embedding_hub.register_vocab(ConcatInputer.vocab)
        self.embedding_hub.register_vocab(FlattenSeqInputer.vocab)
        if self.model.config.use_item_content:
            self.embedding_hub.register_depot(self.item_hub)
            lm_col = self.data.item.lm_col or 'title'
            if self.embedding_hub.has_col(lm_col):
                self.embedding_hub.clone_vocab(
                    col_name=NaturalConcatInputer.special_col,
                    clone_col_name=self.data.item.lm_col or 'title'
                )
            else:
                warnings.warn(f'cannot find lm column in item depot, please ensure no natural inputer is used')
        cat_embeddings = self.embedding_hub(ConcatInputer.vocab.name)  # type: nn.Embedding
        cat_embeddings.weight.data[ConcatInputer.PAD] = torch.zeros_like(cat_embeddings.weight.data[ConcatInputer.PAD])

        # user plugin initialization
        user_plugin = None
        if self.data.user.plugin:
            user_plugin = UserPlugin(
                depot=DepotHub.get(self.data.user.plugin),
                hidden_size=self.model.config.hidden_size,
                select_cols=self.data.user.plugin_cols,
            )

        # legommender initialization
        # self.legommender = self.legommender_class(
        self.legommender = Legommender(
            meta=self.legommender_meta,
            status=self.status,
            config=self.legommender_config,
            column_map=self.column_map,
            embedding_manager=self.embedding_hub,
            user_hub=self.hubs.a_hub(),
            item_hub=self.item_hub,
            user_plugin=user_plugin,
        )
        self.resampler = Resampler(
            legommender=self.legommender,
            item_hub=self.item_hub,
            status=self.status,
        )

        if self.legommender_config.use_neg_sampling:
            self.depots.negative_filter(self.column_map.label_col)

        if self.exp.policy.use_cache:
            for depot in self.depots.depots.values():
                depot.start_caching()

        # data sets initialization
        self.sets = DataSets(hubs=self.hubs, resampler=self.resampler)

    def parse_mode(self):
        modes = set(self.exp.mode.lower().split('_'))
        if Phases.train in modes:
            modes.add(Phases.dev)
        return modes

    def get_loader(self, phase):
        return DataLoader(
            resampler=self.resampler,
            user_set=self.sets.user_set,
            dataset=self.sets[phase],
            shuffle=phase == Phases.train,
            batch_size=self.exp.policy.batch_size,
            pin_memory=self.exp.policy.pin_memory,
            num_workers=5,
            # collate_fn=self.stacker,
        )
