import os
import random
from typing import cast

import pandas as pd
from unitok import BertTokenizer, TransformersTokenizer, EntityTokenizer
from unitok.tokenizer.glove_tokenizer import GloVeTokenizer

from embedder.glove_embedder import GloVeEmbedder
from processor.base_processor import BaseProcessor, Interactions
from utils.config_init import ModelInit


class MINDProcessor(BaseProcessor):
    IID_COL = 'nid'
    UID_COL = 'uid'
    HIS_COL = 'history'
    LBL_COL = 'click'
    NEG_COL = 'neg'

    NEG_TRUNCATE = 100
    REQUIRE_STRINGIFY = False

    @property
    def attrs(self) -> dict:
        return dict(
            title=50,
            abstract=200,
            category=0,
            subcategory=0,
        )

    def config_item_tokenization(self):
        bert_tokenizer = BertTokenizer(vocab='bert')
        llama1_tokenizer = TransformersTokenizer(vocab='llama1', key=ModelInit.get('llama1'))
        glove_tokenizer = GloVeTokenizer(vocab=GloVeEmbedder.get_glove_vocab())

        self.add_item_tokenizer(bert_tokenizer)
        self.add_item_tokenizer(llama1_tokenizer)
        self.add_item_tokenizer(glove_tokenizer)

        self.item.add_feature(tokenizer=EntityTokenizer(vocab='category'), column='category')
        self.item.add_feature(tokenizer=EntityTokenizer(vocab='subcategory'), column='subcategory')

    def _load_items(self, path: str) -> pd.DataFrame:
        return pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep='\t',
            names=[self.IID_COL, 'category', 'subcategory', 'title', 'abstract', 'url', 'tit_ent', 'abs_ent'],
            usecols=[self.IID_COL, 'category', 'subcategory', 'title', 'abstract'],
        )

    def load_items(self) -> pd.DataFrame:
        train_df = self._load_items(os.path.join(self.data_dir, 'train', 'news.tsv'))
        valid_df = self._load_items(os.path.join(self.data_dir, 'dev', 'news.tsv'))
        item_df = pd.concat([train_df, valid_df]).drop_duplicates([self.IID_COL])
        item_df['abstract'] = item_df['abstract'].fillna('')
        item_df['prompt'] = 'Here is a piece of news article. '
        item_df['prompt_title'] = 'Title: '
        item_df['prompt_abstract'] = 'Abstract: '
        item_df['prompt_category'] = 'Category: '
        item_df['prompt_subcategory'] = 'Subcategory: '
        return item_df

    def _load_users(self, path: str) -> pd.DataFrame:
        return pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep='\t',
            names=['imp', self.UID_COL, 'time', self.HIS_COL, 'predict'],
            usecols=[self.UID_COL, self.HIS_COL]
        )

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.item_df[self.IID_COL].unique())

        train_df = self._load_users(os.path.join(self.data_dir, 'train', 'behaviors.tsv'))
        valid_df = self._load_users(os.path.join(self.data_dir, 'dev', 'behaviors.tsv'))
        users = pd.concat([train_df, valid_df]).drop_duplicates([self.UID_COL])
        users[self.HIS_COL] = users[self.HIS_COL].str.split()
        users = users.dropna(subset=[self.HIS_COL])

        users[self.HIS_COL] = users[self.HIS_COL].apply(lambda x: [item for item in x if item in item_set])
        users = users[users[self.HIS_COL].map(lambda x: len(x) > 0)]
        return users

    def _load_interactions(self, path):
        user_set = set(self.user_df[self.UID_COL].unique())

        interactions = pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep='\t',
            names=['imp', self.UID_COL, 'time', self.HIS_COL, 'predict'],
            usecols=[self.UID_COL, 'predict']
        )
        interactions = interactions[interactions[self.UID_COL].isin(user_set)]
        interactions['predict'] = interactions['predict'].str.split().apply(
            lambda x: [item.split('-') for item in x]
        )
        interactions = interactions.explode('predict')
        interactions[[self.IID_COL, self.LBL_COL]] = pd.DataFrame(interactions['predict'].tolist(),
                                                                  index=interactions.index)
        interactions.drop(columns=['predict'], inplace=True)
        interactions[self.LBL_COL] = interactions[self.LBL_COL].astype(int)
        return interactions

    def load_interactions(self):
        train_df = self._load_interactions(os.path.join(self.data_dir, 'train', 'behaviors.tsv'))
        test_df = self._load_interactions(os.path.join(self.data_dir, 'dev', 'behaviors.tsv'))

        # group train_df by UID_COL, select 10% users as valid_df
        users = list(train_df[self.UID_COL].unique())
        random.shuffle(users)
        valid_users = set(users[:int(len(users) * 0.1)])
        valid_df = train_df[train_df[self.UID_COL].isin(valid_users)]
        train_df = train_df[~train_df[self.UID_COL].isin(valid_users)]

        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        return Interactions(train_df, valid_df, test_df)

    def after_load_interactions(self):
        df = pd.concat([self.interactions.train_df, self.interactions.valid_df])
        neg_df = df[df[self.LBL_COL] == 0]
        neg_df = neg_df.groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()
        neg_df.columns = [self.UID_COL, self.NEG_COL]
        self.user_df = self.user_df.merge(neg_df, on=self.UID_COL, how='left')
        # set negative samples to empty list if it is NaN
        self.user_df[self.NEG_COL] = self.user_df[self.NEG_COL].apply(lambda x: x if isinstance(x, list) else [])
