import pandas as pd
from UniTok import UniTok, Vocab
from UniTok.tok import EntTok, SeqTok


class MaxRestrictVocab(Vocab):
    def trim_min_max(self, min_count, max_count):
        """
        trim vocab by min frequency
        :return: trimmed tokens
        """
        _trimmed = []

        vocabs = []
        for index in self._counter:
            if min_count <= self._counter[index] <= max_count:
                vocabs.append(self.i2o[index])
            else:
                _trimmed.append(self.i2o[index])

        self.i2o = dict()
        self.o2i = dict()

        self.set_count_mode(False)
        if self.reserved_tokens is not None:
            self.reserve(self.reserved_tokens)
        self.extend(vocabs)

        self._stable_mode = True
        return _trimmed


session_path = "/data8T_1/qijiong/Data/Goodreads/session.csv"
filtered_session_path = "/data8T_1/qijiong/Data/Goodreads/filtered_session.csv"

df = pd.read_csv(
    filepath_or_buffer=session_path,
    sep='\t',
    header=None,
    names=['uid', 'history', 'neg'],
)

df['history'] = df['history'].apply(lambda x: x.split(' '))
df['neg'] = df['neg'].apply(lambda x: x.split(' ') if isinstance(x, str) and x else [])

n_round = 1
last_vocab_size = None
max_seq_len = 30000

while True:
    print('\n' * 5)
    print(f'------------------ ROUND {n_round} ------------------')
    print(f'Sample size: {len(df)} | Max seq len: {max_seq_len} | Vocab size: {last_vocab_size}')
    n_round += 1

    book_vocab = MaxRestrictVocab(name='bid')

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

    book_vocab.trim_min_max(min_count=5, max_count=10000)

    if last_vocab_size == len(book_vocab):
        if max_seq_len == 100:
            break

        max_seq_len = int(max_seq_len * 0.2)
        if max_seq_len < 100:
            max_seq_len = 100

    last_vocab_size = len(book_vocab)

    df['history'] = df['history'].apply(lambda x: [bid for bid in x if bid in book_vocab.o2i][:max_seq_len])
    df['neg'] = df['neg'].apply(lambda x: [bid for bid in x if bid in book_vocab.o2i][:max_seq_len])

    # if history is empty, drop the row
    df = df[df['history'].apply(lambda x: len(x) > 0)]


df.to_csv(filtered_session_path, sep='\t', index=False, header=False)
