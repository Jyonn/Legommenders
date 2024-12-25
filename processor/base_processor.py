import abc
import os.path
from typing import Optional

import pandas as pd
from pigmento import pnt
from unitok import Vocab, UniTok, EntityTokenizer, EntitiesTokenizer, DigitTokenizer, Symbol


class Interactions(dict):
    train = Symbol('train')
    valid = Symbol('valid')
    test = Symbol('test')
    modes = [train, valid, test]

    def __init__(self, train, valid, test):
        super(Interactions, self).__init__({
            self.train: train,
            self.valid: valid,
            self.test: test,
        })

        if isinstance(train, pd.DataFrame):
            self.train_df, self.valid_df, self.test_df = train, valid, test
            self.train_ut, self.valid_ut, self.test_ut = None, None, None
        else:
            self.train_ut, self.valid_ut, self.test_ut = train, valid, test
            self.train_df, self.valid_df, self.test_df = None, None, None


class BaseProcessor(abc.ABC):
    IID_COL: str
    UID_COL: str
    HIS_COL: str
    LBL_COL: str

    IID_JOB = 'item_id'
    UID_JOB = 'user_id'
    HIS_JOB = 'history'
    LBL_JOB = 'click'

    MAX_HISTORY_PER_USER: int = 100
    REQUIRE_STRINGIFY: bool

    BASE_STORE_DIR = 'data'

    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.save_dir = os.path.join(self.BASE_STORE_DIR, self.get_name())
        os.makedirs(self.save_dir, exist_ok=True)

        self.item_save_dir = os.path.join(self.save_dir, 'items')
        self.user_save_dir = os.path.join(self.save_dir, 'users')

        self._loaded: bool = False

        self.item: Optional[UniTok] = None
        self.user: Optional[UniTok] = None
        self.interactions: Optional[Interactions] = None

        self.item_df: Optional[pd.DataFrame] = None
        self.user_df: Optional[pd.DataFrame] = None

    def get_save_dir(self, mode: Symbol):
        return os.path.join(self.save_dir, mode.name)

    @property
    def default_attrs(self) -> list:
        raise NotImplemented

    @classmethod
    def get_name(cls):
        return cls.__name__.replace('Processor', '').lower()

    def config_item_tokenization(self):
        raise NotImplemented

    def load_items(self) -> pd.DataFrame:
        raise NotImplemented

    def load_users(self) -> pd.DataFrame:
        raise NotImplemented

    def load_interactions(self) -> Interactions:
        raise NotImplemented

    def _stringify(self, df: pd.DataFrame):
        if df is None:
            return None
        if not self.REQUIRE_STRINGIFY:
            return df
        if self.IID_COL in df.columns:
            df[self.IID_COL] = df[self.IID_COL].astype(str)
        if self.UID_COL in df.columns:
            df[self.UID_COL] = df[self.UID_COL].astype(str)
        return df

    @staticmethod
    def require_stringify(func):
        def wrapper(self, *args, **kwargs):
            dfs = func(self, *args, **kwargs)
            if isinstance(dfs, pd.DataFrame):
                return self._stringify(dfs)
            assert isinstance(dfs, tuple)
            return (self._stringify(df) for df in dfs)
        return wrapper

    def load(self):
        pnt(f'load {self.get_name()} processor')

        if os.path.exists(self.item_save_dir) and os.path.exists(self.user_save_dir) and \
                all([os.path.exists(self.get_save_dir(mode)) for mode in Interactions.modes]):
            self.item = UniTok.load(self.item_save_dir)
            pnt(f'loaded {len(self.item)} items')
            self.user = UniTok.load(self.user_save_dir)
            pnt(f'loaded {len(self.user)} users')
            uts = [UniTok.load(self.get_save_dir(mode)) for mode in Interactions.modes]
            for mode, ut in zip(Interactions.modes, uts):
                pnt(f'loaded {len(ut)} {mode.name} interactions')
            self.interactions = Interactions(*uts)
            return

        self.item_df = self.load_items()
        pnt(f'loaded {len(self.item_df)} items')

        self.user_df = self.load_users()
        pnt(f'loaded {len(self.user_df)} users')

        self.interactions = self.load_interactions()
        for mode in Interactions.modes:
            pnt(f'loaded {len(self.interactions[mode])} {mode.name} interactions')

        user_vocab = Vocab(name=self.UID_COL)
        user_vocab.extend(list(self.user_df[self.UID_COL]))
        user_vocab.counter.activate()
        for df in self.interactions.values():
            user_vocab.extend(list(df[self.UID_COL]))
        used_users = user_vocab.counter.trim(min_count=1)
        used_users = set([user_vocab[u] for u in used_users])

        self.user_df = self.user_df[self.user_df[self.UID_COL].isin(used_users)]
        self.user_df = self.user_df.reset_index(drop=True)
        pnt(f'compressed to {len(self.user_df)} users')

        item_vocab = Vocab(name=self.IID_COL)
        item_vocab.extend(list(self.item_df[self.IID_COL]))
        item_vocab.counter.activate()
        for df in self.interactions.values():
            item_vocab.extend(list(df[self.IID_COL]))
        self.user_df[self.HIS_COL].map(item_vocab.extend)
        used_items = item_vocab.counter.trim(min_count=1)
        used_items = set([item_vocab[i] for i in used_items])
        self.item_df = self.item_df[self.item_df[self.IID_COL].isin(used_items)]
        self.item_df = self.item_df.reset_index(drop=True)
        pnt(f'compressed to {len(self.item_df)} items')

        with UniTok() as self.item:
            item_vocab = Vocab(name=self.IID_COL)
            self.item.add_job(tokenizer=EntityTokenizer(vocab=item_vocab), column=self.IID_COL, name=self.IID_JOB, key=True)
            self.config_item_tokenization()
            self.item.tokenize(self.item_df).save(self.item_save_dir)
            pnt(f'tokenized {len(self.item)} items')

        with UniTok() as self.user:
            user_vocab = Vocab(name=self.UID_COL)
            self.user.add_job(tokenizer=EntityTokenizer(vocab=user_vocab), column=self.UID_COL, name=self.UID_JOB, key=True)
            self.user.add_job(tokenizer=EntitiesTokenizer(vocab=item_vocab), column=self.HIS_COL, name=self.HIS_JOB, truncate=0)
            self.user.tokenize(self.user_df).save(self.user_save_dir)
            pnt(f'tokenized {len(self.user)} users')

        uts = []
        for mode in Interactions.modes:
            with UniTok() as ut:
                ut.add_index_job()
                ut.add_job(tokenizer=EntityTokenizer(vocab=user_vocab), column=self.UID_COL, name=self.UID_JOB)
                ut.add_job(tokenizer=EntityTokenizer(vocab=item_vocab), column=self.IID_COL, name=self.IID_JOB)
                ut.add_job(tokenizer=DigitTokenizer(vocab=self.LBL_COL, vocab_size=2), column=self.LBL_COL, name=self.LBL_JOB)
                ut.tokenize(self.interactions[mode]).save(self.get_save_dir(mode))
                pnt(f'tokenized {len(ut)} {mode.name} interactions')

            uts.append(ut)

        self.interactions = Interactions(*uts)
