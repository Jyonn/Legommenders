import torch
from pigmento import pnt
from torch import nn
from torch.nn.modules.module import T

from loader.env import Env
from model.common.base_module import BaseModule
from model.lego_config import LegoConfig
from model.operators.base_llm_operator import BaseLLMOperator
from loader.cacher.repr_cacher import ReprCacher
from loader.column_map import ColumnMap
from utils.shaper import Shaper


class Legommender(BaseModule):
    def __init__(
            self,
            config: LegoConfig,
    ):
        super().__init__()

        self.config: LegoConfig = config

        """initializing basic attributes"""
        # self.meta = meta
        self.item_encoder_class = self.config.item_operator_class
        self.user_encoder_class = self.config.user_operator_class
        self.predictor_class = self.config.predictor_class

        self.use_neg_sampling = self.config.use_neg_sampling
        self.neg_count = self.config.neg_count

        self.embedding_hub = self.config.embedding_hub

        self.user_hub = self.config.user_ut
        self.item_hub = self.config.item_ut

        self.cm = self.config.column_map  # type: ColumnMap

        """initializing core components"""
        self.flatten_mode = self.user_encoder_class.flatten_mode
        # self.item_encoder = None
        # if self.config.use_item_content:
        #     self.item_encoder = self.prepare_item_module()
        # self.user_encoder = self.prepare_user_module()
        # self.predictor = self.prepare_predictor()
        # self.mediator = Mediator(self)
        self.item_op = self.config.item_operator
        self.user_op = self.config.user_operator
        self.predictor = self.config.predictor

        """initializing utils"""
        # special cases for llama
        Env.set_llm_cache(False)
        if self.config.use_item_content:
            if isinstance(self.item_op, BaseLLMOperator):
                if self.item_op.config.layer_split:
                    Env.set_llm_cache(True)

        self.shaper = Shaper()
        self.cacher = ReprCacher(self)

        self.loss_func = nn.CrossEntropyLoss() if self.use_neg_sampling else nn.BCEWithLogitsLoss()

    @staticmethod
    def get_sample_size(item_content):
        if isinstance(item_content, torch.Tensor):
            return item_content.shape[0]
        assert isinstance(item_content, dict)
        key = list(item_content.keys())[0]
        return item_content[key].shape[0]

    def get_item_content(self, batch, col):
        if self.cacher.item.cached:
            indices = batch[col]
            shape = indices.shape
            indices = indices.reshape(-1)
            item_repr = self.cacher.item.repr[indices]
            item_repr = item_repr.reshape(*shape, -1)
            # return self.cacher.item.repr[batch[col]]
            return item_repr

        if not Env.llm_cache:
            _shape = None
            item_content = self.shaper.transform(batch[col])  # batch_size, click_size, max_seq_len
            attention_mask = self.item_op.inputer.get_mask(item_content)
            item_content = self.item_op.inputer.get_embeddings(item_content)
        else:
            _shape = batch[col].shape
            item_content = batch[col].reshape(-1)
            attention_mask = None

        sample_size = self.get_sample_size(item_content)
        allow_batch_size = self.config.item_page_size or sample_size
        batch_num = (sample_size + allow_batch_size - 1) // allow_batch_size

        # item_contents = torch.zeros(sample_size, self.config.hidden_size, dtype=torch.float).to(Setting.device)
        item_contents = self.item_op.get_full_placeholder(sample_size).to(Env.device)
        for i in range(batch_num):
            start = i * allow_batch_size
            end = min((i + 1) * allow_batch_size, sample_size)
            mask = None if attention_mask is None else attention_mask[start:end]
            content = self.item_op(item_content[start:end], mask=mask)
            # print(Structure().analyse_and_stringify(content))
            item_contents[start:end] = content

        if not Env.llm_cache:
            item_contents = self.shaper.recover(item_contents)
        else:
            item_contents = item_contents.view(*_shape, -1)
        return item_contents

    def get_user_content(self, batch):
        if self.cacher.user.cached:
            return self.cacher.user.repr[batch[self.cm.user_col]]

        if self.config.use_item_content and not self.flatten_mode:
            clicks = self.get_item_content(batch, self.cm.history_col)
        else:
            clicks = self.user_op.inputer.get_embeddings(batch[self.cm.history_col])

        user_embedding = self.user_op(
            clicks,
            mask=batch[self.cm.mask_col].to(Env.device),
        )
        return user_embedding

    def forward(self, batch):
        if isinstance(batch[self.cm.item_col], torch.Tensor) and batch[self.cm.item_col].dim() == 1:
            batch[self.cm.item_col] = batch[self.cm.item_col].unsqueeze(1)

        if self.config.use_item_content:
            item_embeddings = self.get_item_content(batch, self.cm.item_col)
        else:
            item_embeddings = self.embedding_hub(self.cm.history_col)(batch[self.cm.item_col].to(Env.device))
        # print(item_embeddings.shape)

        user_embeddings = self.get_user_content(batch)
        # print(user_embeddings.shape)

        if self.use_neg_sampling:
            scores = self.predict_for_neg_sampling(item_embeddings, user_embeddings)
            labels = torch.zeros(scores.shape[0], dtype=torch.long, device=Env.device)
        else:
            scores = self.predict_for_ranking(item_embeddings, user_embeddings)
            labels = batch[self.cm.label_col].float().to(Env.device)

        if Env.is_testing or (Env.is_evaluating and not Env.simple_dev):
            return scores

        return self.loss_func(scores, labels)

    def predict_for_neg_sampling(self, item_embeddings, user_embeddings):
        batch_size, candidate_size, hidden_size = item_embeddings.shape
        if self.predictor.keep_input_dim:
            return self.predictor(user_embeddings, item_embeddings)
        # if user_embeddings: B, S, D
        # user_embeddings = user_embeddings.unsqueeze(1).repeat(1, candidate_size, 1)  # B, K+1, D
        # user_embeddings = user_embeddings.view(-1, hidden_size)
        user_embeddings = self.user_op.prepare_for_predictor(user_embeddings, candidate_size)
        item_embeddings = item_embeddings.view(-1, hidden_size)
        scores = self.predictor(user_embeddings, item_embeddings)
        scores = scores.view(batch_size, -1)
        return scores

    def predict_for_ranking(self, item_embeddings, user_embeddings):
        item_embeddings = item_embeddings.squeeze(1)
        scores = self.predictor(user_embeddings, item_embeddings)
        return scores

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()

    # def prepare_user_module(self):
    #     user_config = self.user_encoder_class.config_class(**combine_config(
    #         config=self.config.user_config,
    #         hidden_size=self.item_encoder.output_dim,
    #         embed_hidden_size=self.config.item_hidden_size,
    #         input_dim=self.config.hidden_size,
    #     ))
    #
    #     if self.flatten_mode:
    #         user_config.inputer_config['item_ut'] = self.config.item_ut
    #         user_config.inputer_config['item_inputs'] = self.config.item_inputs
    #
    #     return self.user_encoder_class(
    #         target_user=True,
    #         lego_config=self.config,
    #     )
    #
    # def prepare_item_module(self):
    #     item_config = self.item_encoder_class.config_class(**combine_config(
    #         config=self.config.item_config,
    #         hidden_size=self.config.hidden_size,
    #         embed_hidden_size=self.config.item_hidden_size,
    #         input_dim=self.config.item_hidden_size,
    #     ))
    #
    #     return self.item_encoder_class(
    #         target_user=False,
    #         lego_config=self.config,
    #     )
    #
    # def prepare_predictor(self):
    #     if self.config.use_neg_sampling and not self.predictor_class.allow_matching:
    #         raise ValueError(f'{self.predictor_class.__name__} does not support negative sampling')
    #
    #     if not self.config.use_neg_sampling and not self.predictor_class.allow_ranking:
    #         raise ValueError(f'{self.predictor_class.__name__} only supports negative sampling')
    #
    #     predictor_config = self.predictor_class.config_class(**combine_config(
    #         config=self.config.predictor_config,
    #         hidden_size=self.config.hidden_size,
    #         embed_hidden_size=self.config.item_hidden_size,
    #     ))
    #
    #     return self.predictor_class(
    #         config=predictor_config,
    #         lego_config=self.config,
    #     )

    def get_parameters(self):
        pretrained_parameters = []
        other_parameters = []
        pretrained_signals = self.item_op.get_pretrained_parameter_names()

        pretrained_names = []
        other_names = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_pretrained = False
            for pretrained_name in pretrained_signals:
                if name.startswith(f'item_encoder.{pretrained_name}'):
                    pretrained_names.append((name, param.data.shape))
                    pretrained_parameters.append(param)
                    is_pretrained = True
                    break

            if not is_pretrained:
                # pnt(f'[N] {name} {param.data.shape}')
                other_names.append((name, param.data.shape))
                other_parameters.append(param)

        for name, shape in pretrained_names:
            pnt(f'[P] {name} {shape}')
        for name, shape in other_names:
            pnt(f'[N] {name} {shape}')

        return pretrained_parameters, other_parameters
