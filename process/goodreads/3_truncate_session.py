import random

import pandas as pd
from UniTok import UniTok, Vocab
from UniTok.tok import EntTok, SplitTok
from tqdm import tqdm

session_path = "/data8T_1/qijiong/Data/Goodreads/session.csv"
truncate_session_path = "/data8T_1/qijiong/Data/Goodreads/truncate_session.csv"

df = pd.read_csv(
    filepath_or_buffer=session_path,
    sep='\t',
    header=None,
    names=['uid', 'history', 'neg'],
)

with open(truncate_session_path, 'w') as f:
    for uid, history, neg in tqdm(zip(df['uid'], df['history'], df['neg'])):
        history = [book_id for book_id in history.split(' ')][:50]
        neg = [book_id for book_id in (neg.split(' ') if isinstance(neg, str) and neg else [])][:50]
        f.write(f"{uid}\t{' '.join(history)}\t{' '.join(neg)}\n")
