import json

import numpy as np
import pandas as pd
from UniTok import UniTok, Vocab
from UniTok.tok import EntTok, SeqTok


filtered_session_path = "/data8T_1/qijiong/Data/Goodreads/filtered_session.csv"
removed_user_path = "/data8T_1/qijiong/Data/Goodreads/filtered_session.csv"

df = pd.read_csv(
    filepath_or_buffer=filtered_session_path,
    sep='\t',
    header=None,
    names=['uid', 'history', 'neg'],
)

df['history'] = df['history'].apply(eval)
df['neg'] = df['neg'].apply(eval)

n_round = 1
last_vocab_size = None

while True:
    print('\n' * 5)
    print(f'------------------ ROUND {n_round} ------------------')
    print(f'Sample size: {len(df)} | Vocab size: {last_vocab_size}')
    n_round += 1

    book_vocab = Vocab(name='bid')

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

    book_vocab.trim(min_count=5)

    if last_vocab_size == len(book_vocab):
        count = input('randomly remove users? (count): ')
        count = int(count)
        if count == 0:
            break

        count = len(df) - count
        indices = np.random.choice(len(df), count, replace=False)
        df = df.iloc[indices]
        continue

    last_vocab_size = len(book_vocab)

    df['history'] = df['history'].apply(lambda x: [bid for bid in x if bid in book_vocab.o2i])
    df['neg'] = df['neg'].apply(lambda x: [bid for bid in x if bid in book_vocab.o2i])

    # if history is empty, drop the row
    df = df[df['history'].apply(lambda x: len(x) >= 10)]


df.to_csv(filtered_session_path, sep='\t', index=False, header=False)
