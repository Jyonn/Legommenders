import os.path

import pandas as pd
from UniTok import UniTok, Column, Vocab, UniDep
from UniTok.tok import BertTok, IdTok, EntTok, SeqTok, NumberTok
from tqdm import tqdm


class Processor:
    def __init__(self, data_dir, store_dir):
        self.data_dir = data_dir
        self.store_dir = store_dir

        self.news_path = os.path.join(self.data_dir, 'articles.parquet')

        self.nid = Vocab(name='nid')
        self.uid = Vocab(name='uid')

    @staticmethod
    def category_handler(s):
        categories = """
        webtv
        om_ekstra_bladet
        abonnement
        services
        tilavis
        eblive
        horoskoper
        webmaster-test-sektion
        migration_catalog
        podcast
        rssfeed
        """.strip().split('\n')
        categories = [c.strip() for c in categories]
        if s in categories:
            s = 'nyheder'
        return s

    def read_news(self):
        df = pd.read_parquet(self.news_path)
        df = df[['article_id', 'title', 'subtitle', 'body', 'category_str']]
        df.columns = ['nid', 'title', 'subtitle', 'body', 'category']
        return df

    def read_user(self, mode='train'):
        df = pd.read_parquet(os.path.join(self.data_dir, mode, 'history.parquet'))
        df = df[['user_id', 'article_id_fixed']]
        df.columns = ['uid', 'history']
        # uid 123 -> train_123
        df['uid'] = df['uid'].apply(lambda x: f'{mode}_{x}')
        # filter out items in the history that are not in the news
        df['history'] = df['history'].apply(lambda x: [str(i) for i in x if str(i) in self.nid.o2i])
        return df

    def read_inters(self, mode='train'):
        columns = ['impression_id', 'user_id', 'article_ids_inview']
        if mode != 'test':
            columns.append('article_ids_clicked')
        df = pd.read_parquet(
            path=os.path.join(self.data_dir, mode, 'behaviors.parquet'),
            columns=columns
        )
        if mode == 'test':
            df.columns = ['imp', 'uid', 'interactions']
            df['click'] = 0
            df['click'] = df['click'].apply(lambda x: [])
        else:
            df.columns = ['imp', 'uid', 'interactions', 'click']
        df['uid'] = df['uid'].apply(lambda x: f'{mode}_{x}')

        data = dict(imp=[], uid=[], nid=[], click=[])
        neg_dict = dict()
        for line in tqdm(df.itertuples(), total=len(df)):
            interactions = list(map(str, line.interactions))
            clicks = list(map(str, line.click))
            data['imp'].extend([line.imp] * len(interactions))
            data['uid'].extend([line.uid] * len(interactions))
            for interaction in interactions:
                data['nid'].append(interaction)
                data['click'].append(int(interaction in clicks))

            if mode == 'test':
                neg_dict[line.uid] = {self.nid.i2o[0]}
                continue

            if line.uid not in neg_dict:
                neg_dict[line.uid] = set()
            neg_samples = [inter for inter in interactions if (inter not in clicks) and (inter in self.nid.o2i)]
            neg_dict[line.uid].update(neg_samples)

        neg_data = dict(uid=[], neg=[])
        for uid, neg_samples in neg_dict.items():
            neg_data['uid'].append(uid)
            neg_data['neg'].append(list(neg_samples))

        return pd.DataFrame(data), pd.DataFrame(neg_data)

    def get_news_tok(self, max_title_len=0, max_subtitle_len=0, max_body_len=0):
        text_tok = BertTok(name='bert', vocab_dir='google-bert/bert-base-multilingual-cased')
        cat_tok = EntTok(name='category')
        cat_tok.pre_handler = self.category_handler
        return UniTok().add_col(Column(
            tok=IdTok(vocab=self.nid),
        )).add_col(Column(
            name='title',
            tok=text_tok,
            max_length=max_title_len,
        )).add_col(Column(
            name='subtitle',
            tok=text_tok,
            max_length=max_subtitle_len,
        )).add_col(Column(
            name='body',
            tok=text_tok,
            max_length=max_body_len,
        )).add_col(Column(
            name='category',
            tok=cat_tok,
        ))

    def get_user_tok(self, max_history: int = 0):
        return UniTok().add_col(Column(
            tok=IdTok(vocab=self.uid),
        )).add_col(Column(
            name='history',
            tok=SeqTok(vocab=self.nid),
            max_length=max_history,
            slice_post=True,
        ))

    def get_inter_tok(self):
        return UniTok().add_index_col(
            name='index'
        ).add_col(Column(
            name='imp',
            tok=EntTok,
        )).add_col(Column(
            tok=EntTok(vocab=self.uid),
        )).add_col(Column(
            tok=EntTok(vocab=self.nid),
        )).add_col(Column(
            tok=NumberTok(name='click', vocab_size=2)
        ))

    def get_neg_tok(self, max_neg: int = 0):
        return UniTok().add_col(Column(
            tok=IdTok(vocab=self.uid),
        )).add_col(Column(
            name='neg',
            tok=SeqTok(vocab=self.nid),
            max_length=max_neg,
            slice_post=True,
        ))

    def tokenize(self, load_news=False, load_user=False):
        if load_news:
            news = UniDep(os.path.join(self.store_dir, 'news'))
            self.nid = news.vocs['nid'].vocab
            print('loaded news from depot')
        else:
            news_df = self.read_news()
            news_tok = self.get_news_tok(
                max_title_len=20,
                max_subtitle_len=60,
                max_body_len=100,
            )
            news_tok.read(news_df).tokenize().store(os.path.join(self.store_dir, 'news'))
            print('news processed')
        self.nid.deny_edit()

        if load_user:
            user = UniDep(os.path.join(self.store_dir, 'user'))
            self.uid = user.vocs['uid'].vocab
            print('loaded user from depot')
        else:
            train_user_df = self.read_user(mode='train')
            valid_user_df = self.read_user(mode='validation')
            test_user_df = self.read_user(mode='test')

            user_df = pd.concat([train_user_df, valid_user_df, test_user_df])
            user_tok = self.get_user_tok(max_history=50)
            user_tok.read(user_df).tokenize().store(os.path.join(self.store_dir, 'user'))
            print('user processed')
        self.uid.deny_edit()

        train_inter_df, train_neg_df = self.read_inters(mode='train')
        inter_tok = self.get_inter_tok()
        inter_tok.read(train_inter_df).tokenize().store(os.path.join(self.store_dir, 'train'))

        valid_inter_df, valid_neg_df = self.read_inters(mode='validation')
        inter_tok = self.get_inter_tok()
        inter_tok.read(valid_inter_df).tokenize().store(os.path.join(self.store_dir, 'valid'))

        test_inter_df, test_neg_df = self.read_inters(mode='test')
        inter_tok = self.get_inter_tok()
        inter_tok.read(test_inter_df).tokenize().store(os.path.join(self.store_dir, 'test'))

        neg_df = pd.concat([train_neg_df, valid_neg_df, test_neg_df])
        neg_tok = self.get_neg_tok(max_neg=250)
        neg_tok.read(neg_df).tokenize().store(os.path.join(self.store_dir, 'neg'))

        # for mode, inter in zip(['train', 'validation', 'test'], [train_inter_df, valid_inter_df, test_inter_df]):
        #     inter_tok = self.get_inter_tok()
        #     inter_tok.read(inter).tokenize().store(os.path.join(self.store_dir, mode))


if __name__ == '__main__':
    processor = Processor(
        data_dir="/home/data4/qijiong/Data/EB-NeRD",
        store_dir="../../data/EB-NeRD"
    )
    processor.tokenize(load_news=True, load_user=True)
    # title: 25
    # subtitle: 60
    # body: 1000
