import json

import numpy as np
import pandas as pd
from UniTok import UniTok, Vocab
from UniTok.tok import EntTok, SeqTok


removed_user_path = "/data8T_1/qijiong/Data/Goodreads/filtered_session.csv"
sixth_session_path = "/data8T_1/qijiong/Data/Goodreads/sixth_session.csv"
book_path = "/data8T_1/qijiong/Data/Goodreads/goodreads_book_works.json"

allowed_books = set()
with open(book_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['original_title'].strip() == '':
            continue
        allowed_books.add(data['best_book_id'])


class RandomlyRemoveVocab(Vocab):
    def randomly_remove(self, count):
        count = len(self) - count
        print('remaining vocab size:', count)
        indices = np.random.choice(len(self), count, replace=False)

        vocabs = []
        for index in indices:
            vocabs.append(self.i2o[index])

        assert len(vocabs) == count
        return vocabs

df = pd.read_csv(
    filepath_or_buffer=removed_user_path,
    sep='\t',
    header=None,
    names=['uid', 'history', 'neg'],
)

df['history'] = df['history'].apply(eval)
df['neg'] = df['neg'].apply(eval)

n_round = 1
last_vocab_size = None
min_count, max_count = 5, 10000

while True:
    print('\n' * 5)
    print(f'------------------ ROUND {n_round} ------------------')
    print(f'Sample size: {len(df)} | Vocab size: {last_vocab_size}')
    n_round += 1

    book_vocab = RandomlyRemoveVocab(name='bid')

    ut = UniTok()
    ut.add_index_col()
    ut.add_col(
        col='uid',
        tok=EntTok,
    ).add_col(
        col='history',
        tok=SeqTok(
            vocab=book_vocab,
        )
    ).add_col(
        col='neg',
        tok=SeqTok(
            vocab=book_vocab,
        )
    )

    ut.read(df).analyse()

    book_vocab.trim(min_count=10)

    if last_vocab_size == len(book_vocab):
        count = input('randomly remove items? (count): ')
        count = int(count)
        item_not_changed = count == 0
        if not item_not_changed:
            vocabs = book_vocab.randomly_remove(count)
            vocabs = set(vocabs)
            vocabs.intersection_update(allowed_books)
            vocabs = list(vocabs)
            book_vocab = Vocab(name='bid')
            book_vocab.extend(vocabs)
            last_vocab_size = len(book_vocab)
            df['history'] = df['history'].apply(lambda x: [bid for bid in x if bid in book_vocab.o2i])
            df['neg'] = df['neg'].apply(lambda x: [bid for bid in x if bid in book_vocab.o2i])
            df = df[df['history'].apply(lambda x: len(x) >= 10)]

        count = input('randomly remove users? (count): ')
        count = int(count)
        if count == 0 and item_not_changed:
            break

        if count == 0:
            continue

        count = len(df) - count
        indices = np.random.choice(len(df), count, replace=False)
        df = df.iloc[indices]
        continue

    last_vocab_size = len(book_vocab)

    df['history'] = df['history'].apply(lambda x: [bid for bid in x if bid in book_vocab.o2i])
    df['neg'] = df['neg'].apply(lambda x: [bid for bid in x if bid in book_vocab.o2i])

    # if history is empty, drop the row
    df = df[df['history'].apply(lambda x: len(x) >= 10)]


df.to_csv(sixth_session_path, sep='\t', index=False, header=False)
