import os

import pandas as pd
from UniTok import UniTok, Vocab, Column
from UniTok.tok import IdTok, BertTok, SeqTok, EntTok, NumberTok


class Processor:
    def __init__(self, store_dir):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

        self.base_path = "/data8T_1/qijiong/Data/Goodreads"
        self.book_vocab, self.user_vocab = Vocab(name='bid'), Vocab(name='uid')

    def read_book_data(self):
        df = pd.read_csv(
            filepath_or_buffer=os.path.join(self.base_path, "book.csv"),
            sep='\t',
        )
        df['bid'] = df['bid'].apply(str)
        return df

    def read_inter_data(self, mode):
        df = pd.read_csv(
            filepath_or_buffer=os.path.join(self.base_path, f"{mode}_inter.csv"),
            sep='\t',
        )
        df['uid'] = df['uid'].apply(str)
        df['bid'] = df['bid'].apply(str)
        return df

    def read_user_data(self):
        df = pd.read_csv(
            filepath_or_buffer=os.path.join(self.base_path, "user.csv"),
            sep='\t',
        )
        df['history'] = df['history'].apply(eval)
        return df

    def read_neg_data(self):
        df = pd.read_csv(
            filepath_or_buffer=os.path.join(self.base_path, "neg.csv"),
            sep='\t',
        )
        df['neg'] = df['neg'].apply(eval)
        return df

    def get_book_tok(self):
        return UniTok().add_col(Column(
            tok=IdTok(vocab=self.book_vocab),
        )).add_col(Column(
            name='title',
            tok=BertTok(name='english'),
            max_length=20,
        ))

    def get_user_tok(self):
        return UniTok().add_col(Column(
            tok=IdTok(vocab=self.user_vocab),
        )).add_col(Column(
            name='history',
            tok=SeqTok(vocab=self.book_vocab)
        ))

    def get_neg_tok(self):
        return UniTok().add_col(Column(
            tok=IdTok(vocab=self.user_vocab),
        )).add_col(Column(
            name='neg',
            tok=SeqTok(vocab=self.book_vocab)
        ))

    def get_inter_tok(self):
        return UniTok().add_index_col(
            name='index'
        ).add_col(Column(
            tok=EntTok(vocab=self.user_vocab),
        )).add_col(Column(
            tok=EntTok(vocab=self.book_vocab),
        )).add_col(Column(
            tok=NumberTok(vocab_size=2, name='click'),
        ))

    def tokenize(self):
        book_data = self.read_book_data()
        book_tok = self.get_book_tok()
        book_tok.read(book_data).tokenize().store_data(os.path.join(self.store_dir, 'book'))

        user_data = self.read_user_data()
        user_tok = self.get_user_tok()
        user_tok.read(user_data).tokenize().store_data(os.path.join(self.store_dir, 'user'))

        neg_data = self.read_neg_data()
        neg_tok = self.get_neg_tok()
        neg_tok.read(neg_data).tokenize().store_data(os.path.join(self.store_dir, 'neg'))

        train_data = self.read_inter_data('train')
        train_tok = self.get_inter_tok()
        train_tok.read(train_data).tokenize().store_data(os.path.join(self.store_dir, 'train'))

        valid_data = self.read_inter_data('dev')
        valid_tok = self.get_inter_tok()
        valid_tok.read(valid_data).tokenize().store_data(os.path.join(self.store_dir, 'dev'))

        test_data = self.read_inter_data('test')
        test_tok = self.get_inter_tok()
        test_tok.read(test_data).tokenize().store_data(os.path.join(self.store_dir, 'test'))

    def tokenize_book(self):
        book_data = self.read_book_data()
        book_tok = self.get_book_tok()
        book_tok.read(book_data).tokenize().store_data(os.path.join(self.store_dir, 'book'))


if __name__ == '__main__':
    processor = Processor(store_dir="../../data/Goodreads")
    # processor.tokenize()
    processor.tokenize_book()
