import json
import os
import random
from datetime import datetime, timezone
from typing import cast

import pandas as pd
from tqdm import tqdm
from unitok import BertTokenizer, TransformersTokenizer
from unitok.tokenizer.glove_tokenizer import GloVeTokenizer

from embedder.glove_embedder import GloVeEmbedder
from processor.base_processor import BaseProcessor, Interactions
from utils.config_init import ModelInit


class GoodreadsProcessor(BaseProcessor):
    IID_COL = 'nid'
    UID_COL = 'uid'
    HIS_COL = 'history'
    LBL_COL = 'click'
    DAT_COL = 'date'

    NUM_TEST = 20_000
    NUM_FINETUNE = 100_000

    REQUIRE_STRINGIFY = False

    @property
    def default_attrs(self):
        return dict(title=50)

    def config_item_tokenization(self):
        bert_tokenizer = BertTokenizer(vocab='bert')
        llama1_tokenizer = TransformersTokenizer(vocab='llama1', key=ModelInit.get('llama1'))
        glove_tokenizer = GloVeTokenizer(vocab=GloVeEmbedder.get_glove_vocab())

        self.add_item_tokenizer(bert_tokenizer)
        self.add_item_tokenizer(llama1_tokenizer)
        self.add_item_tokenizer(glove_tokenizer)

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'goodreads_book_works.json')
        item_df = pd.read_json(path, lines=True)
        item_df = item_df[['best_book_id', 'original_title']]
        # if original title strip is empty, then skip
        item_df = item_df[item_df['original_title'].str.strip() != '']
        item_df.columns = [self.IID_COL, 'title']

        item_df['prompt'] = 'Here is a book. '
        item_df['prompt_title'] = 'Title: '

        return item_df

    @staticmethod
    def _str_to_ts(date_string):
        date_format = "%a %b %d %H:%M:%S %z %Y"
        dt = datetime.strptime(date_string, date_format)
        timestamp = int(dt.replace(tzinfo=timezone.utc).timestamp())
        return timestamp

    def _extract_pos_samples(self, users):
        users = users[users[self.HIS_COL].apply(len) > self.POS_COUNT]

        pos_inters = []
        for index, row in users.iterrows():
            for i in range(self.POS_COUNT):
                pos_inters.append({
                    self.UID_COL: row[self.UID_COL],
                    self.IID_COL: row[self.HIS_COL][-(i + 1)],
                    self.LBL_COL: 1
                })
        self._pos_inters = pd.DataFrame(pos_inters)

        users.loc[:, self.HIS_COL] = users[self.HIS_COL].apply(lambda x: x[-self.MAX_HISTORY_PER_USER - self.POS_COUNT: -self.POS_COUNT])

        return users

    def _load_users(self, interactions):
        item_set = set(self.item_df[self.IID_COL].unique())

        interactions = interactions[interactions[self.IID_COL].isin(item_set)]
        interactions = interactions.groupby(self.UID_COL)
        interactions = interactions.filter(lambda x: x[self.LBL_COL].nunique() == 2)
        self._interactions = interactions

        pos_inters = interactions[interactions[self.LBL_COL] == 1]

        users = pos_inters.sort_values(
            [self.UID_COL, self.DAT_COL]
        ).groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()
        users.columns = [self.UID_COL, self.HIS_COL]

        return self._extract_pos_samples(users)

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.item_df[self.IID_COL].unique())

        path = os.path.join(self.data_dir, 'goodreads_interactions_dedup.json')
        interactions = []
        with open(path, 'r') as f:
            for index, line in tqdm(enumerate(f)):
                data = json.loads(line.strip())
                user_id, book_id, is_read, date = data['user_id'], data['book_id'], data['is_read'], data['date_added']
                interactions.append([user_id, book_id, is_read, date])

        interactions = pd.DataFrame(interactions, columns=[self.UID_COL, self.IID_COL, self.LBL_COL, self.DAT_COL])
        interactions = self._stringify(interactions)
        interactions[self.DAT_COL] = interactions[self.DAT_COL].apply(lambda x: self._str_to_ts(x))
        interactions[self.LBL_COL] = interactions[self.LBL_COL].apply(int)
        interactions = interactions[interactions[self.IID_COL].isin(item_set)]
        return self._load_users(interactions)


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
