from typing import Type

import torch
from torch import nn
from tqdm import tqdm

from loader.global_setting import Setting
from model.common.user_plugin import UserPlugin
from model.operator.base_llm_operator import BaseLLMOperator
from model.operator.base_operator import BaseOperator
from model.utils.column_map import ColumnMap
from loader.embedding.embedding_manager import EmbeddingManager
from model.utils.nr_depot import NRDepot
from utils.printer import printer, Color
from utils.shaper import Shaper
from utils.timer import Timer


class BaseRecommenderConfig:
    def __init__(
            self,
            hidden_size,
            user_config,
            embed_hidden_size=None,
            news_config=None,
            use_news_content: bool = True,
            max_news_content_batch_size: int = 0,
            same_dim_transform: bool = True,
            use_fast_eval: bool = False,
            fast_eval_batch_size: int = 64,
    ):
        self.hidden_size = hidden_size
        self.news_config = news_config
        self.user_config = user_config
        self.use_news_content = use_news_content
        self.embed_hidden_size = embed_hidden_size or hidden_size

        self.max_news_content_batch_size = max_news_content_batch_size
        self.same_dim_transform = same_dim_transform

        self.use_fast_eval = use_fast_eval
        self.fast_eval_batch_size = fast_eval_batch_size

        if self.use_news_content and not self.news_config:
            raise ValueError('news_config is required when use_news_content is True')


class BaseRecommender(nn.Module):
    config_class = BaseRecommenderConfig
    news_encoder_class = None  # type: Type[BaseOperator]
    user_encoder_class = None  # type: Type[BaseOperator]
    use_neg_sampling = True

    def __init__(
            self,
            config: BaseRecommenderConfig,
            column_map: ColumnMap,
            embedding_manager: EmbeddingManager,
            user_nrd: NRDepot,
            news_nrd: NRDepot,
            user_plugin: UserPlugin = None,
    ):
        super().__init__()

        self.config = config  # type: BaseRecommenderConfig
        self.print = printer[(self.__class__.__name__, '|', Color.MAGENTA)]

        self.timer = Timer(activate=False)

        self.column_map = column_map  # type: ColumnMap
        self.embedding_manager = embedding_manager
        self.embedding_table = embedding_manager.get_table()

        self.user_col = column_map.user_col
        self.clicks_col = column_map.clicks_col
        self.candidate_col = column_map.candidate_col
        self.label_col = column_map.label_col
        self.clicks_mask_col = column_map.clicks_mask_col

        self.user_config = self.user_encoder_class.config_class(
            hidden_size=config.hidden_size,
            embed_hidden_size=config.embed_hidden_size,
            **config.user_config
        )
        self.user_encoder = self.user_encoder_class(
            config=self.user_config,
            nrd=user_nrd,
            embedding_manager=embedding_manager
        )
        if self.config.use_news_content:
            self.news_config = self.news_encoder_class.config_class(
                hidden_size=config.hidden_size,
                embed_hidden_size=config.embed_hidden_size,
                **config.news_config
            )
            self.news_encoder = self.news_encoder_class(
                config=self.news_config,
                nrd=news_nrd,
                embedding_manager=embedding_manager,
            )

        self.user_plugin = user_plugin
        self.shaper = Shaper()

        # fast evaluation by caching news representations
        self.fast_eval = False
        self.fast_doc_repr = None

        # special cases for llama
        self.llm_skip = False
        if self.config.use_news_content:
            if isinstance(self.news_encoder, BaseLLMOperator):
                if self.news_encoder.config.layer_split:
                    self.llm_skip = True
                    self.print("LLM SKIP")

    def timing(self, activate=True):
        self.timer.activate = activate

    def get_content(self, batch, col):
        if self.fast_eval:
            return self.fast_doc_repr[batch[col]]

        if not self.llm_skip:
            _shape = None
            news_content = self.shaper.transform(batch[col])  # batch_size, click_size, max_seq_len
            attention_mask = self.news_encoder.inputer.get_mask(news_content)
            news_content = self.news_encoder.inputer.get_embeddings(news_content)
        else:
            _shape = batch[col].shape
            news_content = batch[col].reshape(-1)
            attention_mask = None

        allow_batch_size = self.config.max_news_content_batch_size or news_content.shape[0]
        batch_num = (news_content.shape[0] + allow_batch_size - 1) // allow_batch_size

        news_contents = torch.zeros(news_content.shape[0], self.config.hidden_size, dtype=torch.float).to(Setting.device)
        for i in range(batch_num):
            start = i * allow_batch_size
            end = min((i + 1) * allow_batch_size, news_content.shape[0])
            mask = None if attention_mask is None else attention_mask[start:end]
            content = self.news_encoder(news_content[start:end], mask=mask)
            news_contents[start:end] = content

        if not self.llm_skip:
            news_contents = self.shaper.recover(news_contents)
        else:
            news_contents = news_contents.view(*_shape, -1)
        return news_contents

    def fuse_user_plugin(self, batch, user_embedding):
        if self.user_plugin:
            return self.user_plugin(batch[self.user_col], user_embedding)
        return user_embedding

    def forward(self, batch):
        # print(Structure().analyse_and_stringify(batch))
        # exit(0)
        if self.config.use_news_content:
            candidates = self.get_content(batch, self.candidate_col)  # batch_size, candidate_size, embedding_dim
            clicks = self.get_content(batch, self.clicks_col)  # batch_size, click_size, embedding_dim
        else:
            candidates = self.embedding_manager(self.clicks_col)(batch[self.candidate_col].to(Setting.device))
            clicks = self.user_encoder.inputer.get_embeddings(batch[self.clicks_col])

        user_embedding = self.user_encoder(
            clicks,
            mask=batch[self.clicks_mask_col].to(Setting.device),
        )

        self.fuse_user_plugin(batch, user_embedding)

        results = self.predict(user_embedding, candidates, batch)
        # if not Setting.status.is_testing:
        #     results += l2_loss
        return results

    def predict(self, user_embedding, candidates, batch) -> [torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def start_fast_eval(self, doc_list):
        if not self.config.use_news_content:
            return
        if not self.config.use_fast_eval:
            return
        if self.fast_eval:
            return

        self.print("start fast eval")
        self.fast_eval = True
        self.fast_doc_repr = torch.zeros(
            len(doc_list), self.config.hidden_size, dtype=torch.float).to(Setting.device)

        batch_size = self.config.fast_eval_batch_size
        i_batch = 0
        cache_embeddings = []
        cache_masks = []
        current_embeddings = []
        current_masks = []

        with torch.no_grad():
            for index, sample in enumerate(tqdm(doc_list)):
                if self.llm_skip:
                    # embedding = torch.tensor([index], dtype=torch.long)
                    # mask = None
                    current_embeddings.append(index)
                    current_masks = None
                else:
                    mask = self.news_encoder.inputer.get_mask(sample)
                    embedding = self.news_encoder.inputer.get_embeddings(sample)
                    # mask = mask.unsqueeze(0)
                    # embedding = embedding.unsqueeze(0)
                    current_embeddings.append(embedding)
                    current_masks.append(mask)
                # self.fast_doc_repr[index] = self.news_encoder(embedding, mask).squeeze(0)

                i_batch += 1
                if i_batch >= batch_size:
                    cache_embeddings.append(current_embeddings)
                    cache_masks.append(current_masks)
                    current_embeddings = []
                    current_masks = []
                    i_batch = 0

            if i_batch > 0:
                cache_embeddings.append(current_embeddings)
                cache_masks.append(current_masks)

        with torch.no_grad():
            for i, (embeddings, masks) in enumerate(zip(cache_embeddings, cache_masks)):
                current_batch_size = len(embeddings)
                embeddings = torch.stack(embeddings)
                if masks is not None:
                    masks = torch.stack(masks)
                else:
                    masks = None
                embeddings = self.news_encoder(embeddings, masks)
                self.fast_doc_repr[i * batch_size: i * batch_size + current_batch_size] = embeddings

    def end_fast_eval(self):
        self.fast_eval = False
        self.fast_doc_repr = None

    def parameter_split(self):
        news_names, news_parameters = self.news_encoder.get_pretrained_parameters(prefix='news_encoder')
        rec_parameters = self.named_parameters()
        rec_parameters = filter(lambda p: p[1].requires_grad, rec_parameters)
        rec_parameters = filter(lambda p: p[0] not in news_names, rec_parameters)
        rec_parameters = map(lambda p: p[1], rec_parameters)
        return news_parameters, rec_parameters

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()
