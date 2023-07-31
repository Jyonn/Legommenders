import os

import pandas as pd
from UniTok import UniTok, Vocab, Column
from UniTok.tok import IdTok, BaseTok
from transformers import LlamaTokenizer


class LlamaTok(BaseTok):
    return_list = True

    def __init__(self, name, vocab_dir):
        super(LlamaTok, self).__init__(name=name)
        self.tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path=vocab_dir)
        vocab = [self.tokenizer.convert_ids_to_tokens(i) for i in range(self.tokenizer.vocab_size)]
        self.vocab.extend(vocab)

    def t(self, obj) -> [int, list]:
        if pd.notnull(obj):
            ts = self.tokenizer.tokenize(obj)
            ids = self.tokenizer.convert_tokens_to_ids(ts)
        else:
            ids = []
        return ids


class Processor:
    def __init__(self, store_dir):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

        # self.base_path = "/data8T_1/qijiong/Data/Goodreads"
        self.base_path = '/home/qijiong/Data/Goodreads'
        self.book_vocab, self.user_vocab = Vocab(name='bid'), Vocab(name='uid')

    def read_book_data(self):
        df = pd.read_csv(
            filepath_or_buffer=os.path.join(self.base_path, "book.csv"),
            sep='\t',
        )
        df['bid'] = df['bid'].apply(str)
        df['title'] = df['title'].apply(lambda x: 'Book named ' + x)
        return df

    def get_book_tok(self):
        return UniTok().add_col(Column(
            tok=IdTok(vocab=self.book_vocab),
        )).add_col(Column(
            name='title',
            tok=LlamaTok(name='llama', vocab_dir='/home/data1/qijiong/llama-7b'),
            max_length=20,
        ))

    def tokenize(self):
        book_data = self.read_book_data()
        book_tok = self.get_book_tok()
        book_tok.read(book_data).tokenize().store_data(os.path.join(self.store_dir, 'book-llama'))


if __name__ == '__main__':
    processor = Processor(store_dir="../../data/Goodreads")
    processor.tokenize()
