import os
import random
from typing import cast

import pandas as pd
from unitok import BertTokenizer, TransformersTokenizer, EntityTokenizer

from processor.base_processor import BaseProcessor, Interactions


class GoodreadsProcessor(BaseProcessor):
    IID_COL = 'nid'
    UID_COL = 'uid'
    HIS_COL = 'history'
    LBL_COL = 'click'

    NUM_TEST = 20_000
    NUM_FINETUNE = 100_000

    REQUIRE_STRINGIFY = False

    @property
    def default_attrs(self):
        return ['title']

    def config_item_tokenization(self):
        bert_tokenizer = BertTokenizer(vocab='bert')
        llama1_tokenizer = TransformersTokenizer(vocab='llama1', key='huggyllama/llama-7b')
        self.item.add_job(tokenizer=bert_tokenizer, column='title', name='title@bert', truncate=50)
        self.item.add_job(tokenizer=bert_tokenizer, column='abstract', name='abstract@bert', truncate=200)
        self.item.add_job(tokenizer=bert_tokenizer, column='category', name='category@bert', truncate=0)
        self.item.add_job(tokenizer=bert_tokenizer, column='subcategory', name='subcategory@bert', truncate=0)
        self.item.add_job(tokenizer=llama1_tokenizer, column='title', name='title@llama1', truncate=50)
        self.item.add_job(tokenizer=llama1_tokenizer, column='abstract', name='abstract@llama1', truncate=200)
        self.item.add_job(tokenizer=llama1_tokenizer, column='category', name='category@llama1', truncate=0)
        self.item.add_job(tokenizer=llama1_tokenizer, column='subcategory', name='subcategory@llama1', truncate=0)
        self.item.add_job(tokenizer=EntityTokenizer(vocab='category'), column='category')
        self.item.add_job(tokenizer=EntityTokenizer(vocab='subcategory'), column='subcategory')

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'goodreads_book_works.json')
        items = pd.read_json(path, lines=True)
        items = items[['best_book_id', 'original_title']]
        # if original title strip is empty, then skip
        items = items[items['original_title'].str.strip() != '']
        items.columns = [self.IID_COL, 'title']
        return items

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

    def load_interactions(self) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
