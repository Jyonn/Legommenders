import os

import pandas as pd
from UniTok import Vocab, UniTok, Column
from UniTok.tok import IdTok, EntTok, BertTok, SplitTok, NumberTok


class Processor:
    def __init__(self, data_dir, store_dir):
        self.data_dir = data_dir
        self.store_dir = store_dir

        os.makedirs(self.store_dir, exist_ok=True)

        self.nid = Vocab(name='nid')
        self.uid = Vocab(name='uid')

        self.news_df = self.read_news_data()
        self.user_df = self.combine_user_df()

    def read_news_data(self):
        news_file = os.path.join(self.data_dir, 'news.tsv')
        return pd.read_csv(
            filepath_or_buffer=news_file,
            sep='\t',
            header=0,
            names=['nid', 'cat', 'topic', 'title', 'body', 'entity', 'content'],
            usecols=['nid', 'cat', 'topic', 'title', 'body'],
        )

    def read_user_data(self, mode):
        inter_file = os.path.join(self.data_dir, f'{mode}.tsv')
        return pd.read_csv(
            filepath_or_buffer=inter_file,
            sep='\t',
            header=0,
            names=['uid', 'history', 'dwell_time', 'exposure_time', 'pos', 'neg', 'start', 'end', 'dwell_time_pos'],
            usecols=['uid', 'history', 'pos', 'neg'],
        )

    def read_test_data(self):
        return pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, 'personalized_test.tsv'),
            sep='\t',
            header=0,
            names=['uid', 'history', 'pos', 'rewrite_titles'],
            usecols=['uid', 'history', 'pos'],
        )

    def combine_user_df(self):
        user_train_df = self.read_user_data('train')
        user_dev_df = self.read_user_data('valid')

        # tag train df as 0 and dev df as 1
        user_train_df['tag'] = 0
        user_dev_df['tag'] = 1

        user_df = pd.concat([user_train_df, user_dev_df])
        # remove original uid and set new uid
        user_df = user_df.drop(columns=['uid'])
        user_df = user_df.reset_index(drop=True)
        user_df['uid'] = user_df.index
        return user_df

    @staticmethod
    def _get_inter_df(df):
        uid_list = []
        nid_list = []
        clk_list = []

        # iterate over the user df
        for i, row in df.iterrows():
            uid = row['uid']
            pos = row['pos']
            neg = row['neg']

            # add positive interactions
            for nid in pos.split(' '):
                uid_list.append(uid)
                nid_list.append(nid)
                clk_list.append(1)

            # add negative interactions
            for nid in neg.split(' '):
                uid_list.append(uid)
                nid_list.append(nid)
                clk_list.append(0)

        return pd.DataFrame({
            'uid': uid_list,
            'nid': nid_list,
            'click': clk_list,
        })

    def get_inter_df(self):
        inter_dfs = []
        for tag in [0, 1]:
            df = self.user_df[self.user_df['tag'] == tag]
            inter_dfs.append(self._get_inter_df(df))
        return inter_dfs

    def get_news_tok(self, max_title_len, max_body_len):
        bert_tok = BertTok(name='english', vocab_dir='bert-base-uncased')

        return UniTok().add_col(Column(
            tok=IdTok(vocab=self.nid),
        )).add_col(Column(
            name='cat',
            tok=EntTok,
        )).add_col(Column(
            name='topic',
            tok=EntTok,
        )).add_col(Column(
            name='title',
            tok=bert_tok,
            max_length=max_title_len,
        )).add_col(Column(
            name='body',
            tok=bert_tok,
            max_length=max_body_len,
        ))

    def get_user_tok(self, max_history_len):
        return UniTok().add_col(Column(
            tok=IdTok(vocab=self.uid),
        )).add_col(Column(
            name='history',
            tok=SplitTok(
                sep=' ',
                vocab=self.nid,
            ),
            max_length=max_history_len,
            slice_post=True,
        ))

    def get_test_user_tok(self, max_history_len):
        return UniTok().add_col(Column(
            tok=IdTok(vocab=self.uid),
        )).add_col(Column(
            name='history',
            tok=SplitTok(
                sep=',',
                vocab=self.nid,
            ),
            max_length=max_history_len,
            slice_post=True,
        )).add_col(Column(
            name='neg',
            tok=SplitTok(
                sep=',',
                vocab=self.nid,
            ),
            max_length=0,
        ))

    def get_inter_tok(self):
        return UniTok().add_index_col(
            name='index'
        ).add_col(Column(
            tok=EntTok(vocab=self.uid),
        )).add_col(Column(
            tok=EntTok(vocab=self.nid),
        )).add_col(Column(
            tok=NumberTok(name='click', vocab_size=2)
        ))

    def get_neg_tok(self, max_neg_len):
        return UniTok().add_col(Column(
            tok=IdTok(vocab=self.uid),
        )).add_col(Column(
            tok=EntTok(vocab=self.uid),
        )).add_col(Column(
            name='neg',
            tok=SplitTok(
                sep=' ',
                vocab=self.nid,
            ),
            max_length=max_neg_len,
            slice_post=True,
        ))

    def analyse_news(self):
        tok = self.get_news_tok(max_title_len=0, max_body_len=0)
        tok.read(self.news_df).analyse()

    def analyse_user(self):
        tok = self.get_user_tok(max_history_len=0)
        tok.read(self.user_df).analyse()

    def analyse_neg(self):
        tok = self.get_neg_tok(max_neg_len=0)
        tok.read(self.user_df).analyse()

    def tokenize(self):
        news_tok = self.get_news_tok(max_title_len=30, max_body_len=500)
        news_tok.read(self.news_df).tokenize().store(os.path.join(self.store_dir, 'news'))
        # self.nid = Vocab('nid').load(os.path.join(self.store_dir, 'news'))
        # self.nid.deny_edit()

        user_tok = self.get_user_tok(max_history_len=50)
        user_tok.read(self.user_df).tokenize().store(os.path.join(self.store_dir, 'user'))

        neg_tok = self.get_neg_tok(max_neg_len=0)
        neg_tok.read(self.user_df).tokenize().store(os.path.join(self.store_dir, 'neg'))

        train_df, dev_df = self.get_inter_df()
        for mode, df in zip(['train', 'valid'], [train_df, dev_df]):
            inter_tok = self.get_inter_tok()
            inter_tok.read(df).tokenize().store(os.path.join(self.store_dir, mode))

    def tokenize_test(self):
        self.nid = Vocab('nid').load(os.path.join(self.store_dir, 'news'))
        self.nid.deny_edit()

        user_df = self.read_test_data()
        user_df['neg'] = self.nid.i2o[0]
        user_tok = self.get_test_user_tok(max_history_len=50)
        user_tok.read(user_df).tokenize().store(os.path.join(self.store_dir, 'user-test'))
        uid = Vocab('uid').load(os.path.join(self.store_dir, 'user-test'))

        fake_inter_data = pd.DataFrame({
            'uid': [self.uid.i2o[0]],
            'nid': [self.nid.i2o[0]],
            'click': [1],
        })
        inter_tok = self.get_inter_tok()
        inter_tok.read(fake_inter_data).tokenize().store(os.path.join(self.store_dir, 'test'))


if __name__ == '__main__':
    p = Processor(
        data_dir='/home/qijiong/Data/PENS',
        store_dir='../../data/PENS',
    )
    p.tokenize_test()
