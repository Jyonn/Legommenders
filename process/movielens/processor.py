import os
import random

import pandas as pd
from UniTok import Vocab, UniTok, Column
from UniTok.tok import NumberTok, IdTok, SeqTok, BertTok, EntTok


class Processor:
    def __init__(self, data_dir, store_dir):
        self.data_dir = data_dir
        self.item_path = os.path.join(data_dir, 'u.item')
        self.test_path = os.path.join(data_dir, 'ua.test')
        self.train_path = os.path.join(data_dir, 'ua.base')

        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

        self.user_voc = Vocab(name='uid')
        self.item_voc = Vocab(name='mid')

        self.history_df, self.train_df, self.dev_df, self.test_df = self.parse_data()
        self.item_df = self.parse_item()

        self.user_voc.deny_edit()
        self.item_voc.deny_edit()

    def parse_data(self):
        df = pd.read_csv(
            filepath_or_buffer=self.train_path,
            sep='\t',
            header=None,
            names=['uid', 'mid', 'rating', 'timestamp'],
        )

        # remove ratings equal to 3
        df = df[df['rating'] != 3]

        session_dict = dict()  # uid -> positive mid list
        inter_list = []

        for row in df.iterrows():
            uid = str(row[1]['uid'])
            mid = str(row[1]['mid'])

            self.user_voc.append(uid)
            self.item_voc.append(mid)

            rating = row[1]['rating']
            ts = row[1]['timestamp']
            if uid not in session_dict:
                session_dict[uid] = []
            if rating > 3:
                session_dict[uid].append((mid, ts))
            else:
                inter_list.append((uid, mid, 0))

        for uid in session_dict:
            session_dict[uid].sort(key=lambda x: x[1])
            session_dict[uid] = [x[0] for x in session_dict[uid]]
            if len(session_dict[uid]) > 5:
                for i in range(5, len(session_dict[uid])):
                    inter_list.append((uid, session_dict[uid][i], 1))
            session_dict[uid] = session_dict[uid][:5]

        uid_list = []
        history_list = []
        for uid in session_dict:
            uid_list.append(uid)
            history_list.append(session_dict[uid])
        history_df = pd.DataFrame({'uid': uid_list, 'history': history_list})

        # shuffle inter_df and split into train and dev
        random.shuffle(inter_list)
        inter_df = pd.DataFrame(inter_list, columns=['uid', 'mid', 'click'])

        train_size = int(len(inter_list) * 0.8)
        train_df = inter_df[:train_size]
        dev_df = inter_df[train_size:]

        # reset index
        train_df.reset_index(drop=True, inplace=True)
        dev_df.reset_index(drop=True, inplace=True)

        # read test data
        test_df = pd.read_csv(
            filepath_or_buffer=self.test_path,
            sep='\t',
            header=None,
            names=['uid', 'mid', 'rating', 'timestamp'],
        )
        test_df['uid'] = test_df['uid'].apply(lambda x: str(x))
        test_df['mid'] = test_df['mid'].apply(lambda x: str(x))

        test_df = test_df[test_df['rating'] != 3]
        test_df.reset_index(drop=True, inplace=True)

        test_uids = test_df['uid'].tolist()
        test_mids = test_df['mid'].tolist()
        self.user_voc.extend(test_uids)
        self.item_voc.extend(test_mids)

        # add click column based on the rating value
        test_df['click'] = test_df['rating'].apply(lambda x: int(x > 3))

        # select users has both positive and negative samples
        pos_neg_dict = dict()
        for row in test_df.iterrows():
            uid = row[1]['uid']
            click = row[1]['click']
            if uid not in pos_neg_dict:
                pos_neg_dict[uid] = [0, 0]
            pos_neg_dict[uid][click] += 1

        allowed_uids = []
        for uid in pos_neg_dict:
            if pos_neg_dict[uid][0] > 0 and pos_neg_dict[uid][1] > 0:
                allowed_uids.append(uid)

        test_df = test_df[test_df['uid'].isin(allowed_uids)]

        # remove rating and timestamp column
        test_df.drop(columns=['rating', 'timestamp'], inplace=True)

        return history_df, train_df, dev_df, test_df

    def parse_item(self):
        df = pd.read_csv(
            filepath_or_buffer=self.item_path,
            sep='|',
            header=None,
            encoding='ISO-8859-1',
        )
        df = df[[0, 1]]
        df.columns = ['mid', 'name']
        df['mid'] = df['mid'].apply(lambda x: str(x))
        mids = df['mid'].tolist()
        self.item_voc.extend(mids)
        # split name into name and year
        df['year'] = df['name'].apply(lambda x: x[-5:-1])
        df['name'] = df['name'].apply(lambda x: x[:-7])
        return df

    def get_inter_tok(self):
        return UniTok().add_index_col(
            name='index',
        ).add_col(
            col='uid',
            tok=EntTok(vocab=self.user_voc),
        ).add_col(
            col='mid',
            tok=EntTok(vocab=self.item_voc),
        ).add_col(
            col='click',
            tok=NumberTok(vocab_size=2, name='click'),
        )

    def get_history_tok(self):
        return UniTok().add_col(
            col='uid',
            tok=IdTok(vocab=self.user_voc),
        ).add_col(Column(
            name='history',
            tok=SeqTok(vocab=self.item_voc),
            max_length=20,
        ))

    def get_neg_tok(self):
        return UniTok().add_col(
            col='uid',
            tok=IdTok(vocab=self.user_voc),
        ).add_col(Column(
            name='neg',
            tok=SeqTok(vocab=self.item_voc),
        ))

    def get_item_tok(self):
        bert_tok = BertTok(name='bert', vocab_dir='bert-base-uncased')
        return UniTok().add_col(
            col='mid',
            tok=IdTok(vocab=self.item_voc),
        ).add_col(
            col='name',
            tok=bert_tok,
        ).add_col(
            col='year',
            tok=bert_tok,
        )

    def analyse(self):
        self.get_history_tok().read(self.history_df).analyse()

    def tokenize(self):
        self.get_inter_tok().read(self.train_df).tokenize().store(os.path.join(self.store_dir, 'train'))
        self.get_inter_tok().read(self.dev_df).tokenize().store(os.path.join(self.store_dir, 'dev'))
        self.get_inter_tok().read(self.test_df).tokenize().store(os.path.join(self.store_dir, 'test'))

        self.get_history_tok().read(self.history_df).tokenize().store(os.path.join(self.store_dir, 'user'))

    def tokenize_item(self):
        self.get_item_tok().read(self.item_df).tokenize().store(os.path.join(self.store_dir, 'item'))

    def neg_sample(self):
        neg = self.train_df[self.train_df['click'] == 0]
        neg = neg[['uid', 'mid']]
        neg = neg.groupby('uid').agg({'mid': lambda x: list(x)})
        neg_dict = neg.to_dict()['mid']

        for uid in self.user_voc.o2i:
            if uid not in neg_dict:
                neg_dict[uid] = []

        uid_list = []
        neg_list = []
        for uid in neg_dict:
            uid_list.append(uid)
            neg_list.append(neg_dict[uid])

        df = pd.DataFrame({'uid': uid_list, 'neg': neg_list})
        self.get_neg_tok().read(df).tokenize().store(os.path.join(self.store_dir, 'neg'))


if __name__ == '__main__':
    processor = Processor(
        data_dir='/data1/qijiong/Data/MovieLens/ml-100k',
        store_dir='../../data/MovieLens-100k',
    )
    # processor.tokenize()
    # processor.neg_sample()
    processor.tokenize_item()
