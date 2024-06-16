import os.path

import pandas as pd
from UniTok import UniTok, Column, Vocab, UniDep
from UniTok.tok import BertTok, IdTok, EntTok, SeqTok, NumberTok, BaseTok
from tqdm import tqdm
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

        self.news_path = os.path.join(self.data_dir, 'articles.parquet')

        self.nid = Vocab(name='nid')
        self.uid = Vocab(name='uid')

    def read_news(self):
        df = pd.read_parquet(self.news_path)
        df = df[['article_id', 'title', 'subtitle', 'body', 'category_str']]
        df.columns = ['nid', 'title', 'subtitle', 'body', 'category']
        return df

    def get_news_tok(self, max_title_len=0, max_subtitle_len=0, max_body_len=0, max_cat_len=0):
        text_tok = LlamaTok(name='llama', vocab_dir='/home/data1/qijiong/llama-7b')

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
            tok=text_tok,
            max_length=max_cat_len,
        ))

    def tokenize(self):
        news_df = self.read_news()
        news_tok = self.get_news_tok(
            max_title_len=20,
            max_subtitle_len=60,
            max_body_len=100,
            max_cat_len=5,
        )
        news_tok.read(news_df).tokenize().store(os.path.join(self.store_dir, 'news-llama'))
        print('news processed')


if __name__ == '__main__':
    processor = Processor(
        data_dir="/home/data4/qijiong/Data/EB-NeRD",
        store_dir="../../data/EB-NeRD"
    )
    processor.tokenize()
    # title: 25
    # subtitle: 60
    # body: 1000
