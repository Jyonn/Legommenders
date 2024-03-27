import os

import pandas as pd
from UniTok import Vocab, UniTok, Column
from UniTok.tok import IdTok, EntTok, BertTok, SplitTok, NumberTok, BaseTok
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
    def __init__(self, data_dir, store_dir):
        self.data_dir = data_dir
        self.store_dir = store_dir

        os.makedirs(self.store_dir, exist_ok=True)

        self.nid = Vocab(name='nid')
        self.uid = Vocab(name='uid')

        self.news_df = self.read_news_data()

    def read_news_data(self):
        news_file = os.path.join(self.data_dir, 'news.tsv')
        return pd.read_csv(
            filepath_or_buffer=news_file,
            sep='\t',
            header=0,
            names=['nid', 'cat', 'topic', 'title', 'body', 'entity', 'content'],
            usecols=['nid', 'cat', 'topic', 'title', 'body'],
        )

    def get_news_tok(self, max_title_len, max_body_len):
        txt_tok = LlamaTok(name='llama', vocab_dir='/home/data1/qijiong/llama-7b')

        return UniTok().add_col(Column(
            tok=IdTok(vocab=self.nid),
        )).add_col(Column(
            name='cat',
            tok=txt_tok,
        )).add_col(Column(
            name='topic',
            tok=txt_tok,
        )).add_col(Column(
            name='title',
            tok=txt_tok,
            max_length=max_title_len,
        )).add_col(Column(
            name='body',
            tok=txt_tok,
            max_length=max_body_len,
        ))

    def tokenize(self):
        news_tok = self.get_news_tok(max_title_len=30, max_body_len=500)
        news_tok.read(self.news_df).tokenize().store(os.path.join(self.store_dir, 'news-llama'))


if __name__ == '__main__':
    p = Processor(
        data_dir='/home/qijiong/Data/PENS',
        store_dir='../../data/PENS',
    )
    p.tokenize()
