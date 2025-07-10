"""
base_processor.py

A generic **dataset pre-processing scaffold** that
- loads raw CSV/Parquet files (items / users / interactions),
- optionally regenerates the tokenized dataset,
- builds UniTok “vocabulary hubs” for users, items and interactions,
- performs several convenience compressions (remove users / items that
  never appear, truncate negative histories, …).

Concrete datasets must inherit from `BaseProcessor` and implement:
    • class-level constants  (column names, features, …)
    • `load_items`, `load_users`, `load_interactions`
    • (optionally) `config_item_tokenization`,
                   `config_user_tokenization`,
                   `config_inter_tokenization`

The child-class can focus solely on *where* to load the raw files from,
while the heavy lifting (tokenizer wiring, saving, …) is handled here.
"""

from __future__ import annotations

import abc
import os
from typing import Optional

import pandas as pd
from pigmento import pnt
from unitok import (
    Vocab, UniTok,
    EntityTokenizer, EntitiesTokenizer, DigitTokenizer,
    Symbol, VocabularyHub, BaseTokenizer, Feature,
)

# ────────────────────────────────────────────────────────────────────────────
#                              Helper container
# ────────────────────────────────────────────────────────────────────────────
class Interactions(dict):
    """
    Thin wrapper around three DataFrames / UniToks that represent
    TRAIN / VALID / TEST interactions.  Identity comparison (`is`) via
    Symbol objects keeps downstream code neat and typo-free.
    """
    train = Symbol("train")
    valid = Symbol("valid")
    test  = Symbol("test")
    modes = [train, valid, test]

    def __init__(self, train, valid, test):
        super().__init__({
            self.train: train,
            self.valid: valid,
            self.test: test,
        })

        # Store both “raw” (DataFrame) and “tokenized” (UniTok) variants
        if isinstance(train, pd.DataFrame):
            self.train_df, self.valid_df, self.test_df = train, valid, test
            self.train_ut = self.valid_ut = self.test_ut = None
        else:
            self.train_ut, self.valid_ut, self.test_ut = train, valid, test
            self.train_df = self.valid_df = self.test_df = None


# ────────────────────────────────────────────────────────────────────────────
#                              BaseProcessor
# ────────────────────────────────────────────────────────────────────────────
class BaseProcessor(abc.ABC):
    """
    Abstract super-class shared by all dataset processors.

    Sub-classes must define a handful of class attributes describing the
    column layout (IID_COL, UID_COL, …) as well as implement the loader
    functions that fetch raw data from disk / web / db.
    """

    # ---------------- Columns expected in raw csv ----------------------
    IID_COL: str         # item identifier column
    UID_COL: str         # user identifier column
    HIS_COL: str         # column with user click history (list/str)
    LBL_COL: str         # target label column
    NEG_COL: str         # optional column with negative samples

    # ---------------- Logical tokenized feature names ------------------
    IID_FEAT = "item_id"
    UID_FEAT = "user_id"
    HIS_FEAT = "history"
    LBL_FEAT = "click"

    # ---------------- Behaviour toggles --------------------------------
    REQUIRE_STRINGIFY: bool  # whether we must convert IDs to str first
    NEG_TRUNCATE: int        # maximum negatives kept per user (0 = all)

    # ---------------- Disk layout --------------------------------------
    BASE_STORE_DIR = "data"  # <BASE_STORE_DIR>/<dataset_name>/…

    # ===================================================================
    #                           Construction
    # ===================================================================
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir

        # Root path dedicated to this dataset (e.g. data/ml1m/)
        self.save_dir = os.path.join(self.BASE_STORE_DIR, self.get_name())
        os.makedirs(self.save_dir, exist_ok=True)

        # Sub-dirs for the serialised UniTok objects
        self.item_save_dir = os.path.join(self.save_dir, "items")
        self.user_save_dir = os.path.join(self.save_dir, "users")

        # Runtime state toggles / caches
        self._loaded = False

        self.item: Optional[UniTok] = None
        self.user: Optional[UniTok] = None
        self.interactions: Optional[Interactions] = None

        # Raw DataFrames
        self.item_df: Optional[pd.DataFrame] = None
        self.user_df: Optional[pd.DataFrame] = None

    def get_save_dir(self, mode: Symbol):
        return os.path.join(self.save_dir, mode.name)
    
    # ------------------------------------------------------------------
    #                      Abstract interface
    # ------------------------------------------------------------------
    @property
    @abc.abstractmethod
    def attrs(self) -> dict:
        """Mapping attr_name → max_token_len (used for description fields)."""
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        """Infer dataset name from ‘FooProcessor’ → ‘foo’."""
        return cls.__name__.replace("Processor", "").lower()

    # ------------------------------ tokenizer hooks --------------------
    def add_item_tokenizer(self, tokenizer: BaseTokenizer):
        """
        Plug an *additional* tokenizer for textual description fields.
        """
        name = tokenizer.vocab.name
        # prompt@[name], title@[name] … depending on attr list
        self.item.add_feature(tokenizer=tokenizer, column="prompt", name=f"prompt@{name}")
        for attr in self.attrs:
            self.item.add_feature(
                tokenizer=tokenizer,
                column=attr,
                name=f"{attr}@{name}",
                truncate=self.attrs[attr],
            )
            self.item.add_feature(
                tokenizer=tokenizer,
                column=f"prompt_{attr}",
                name=f"prompt_{attr}@{name}",
            )

    # The following three can be overridden to inject extra features
    def config_item_tokenization(self):
        pass

    def config_user_tokenization(self):
        pass

    def config_inter_tokenization(self, ut: UniTok):
        pass

    # ------------------------- raw-data loaders ------------------------
    def load_items(self) -> pd.DataFrame:        ...
    def load_users(self) -> pd.DataFrame:        ...
    def load_interactions(self) -> Interactions: ...

    # ------------------------------------------------------------------
    #                 Helper: ensure ids are strings
    # ------------------------------------------------------------------
    def _stringify(self, df: Optional[pd.DataFrame]):
        if df is None or not self.REQUIRE_STRINGIFY:
            return df
        if self.IID_COL in df.columns:
            df[self.IID_COL] = df[self.IID_COL].astype(str)
        if self.UID_COL in df.columns:
            df[self.UID_COL] = df[self.UID_COL].astype(str)
        return df

    @staticmethod
    def require_stringify(func):
        """
        Decorator that guarantees load_* methods automatically stringify
        IDs when `REQUIRE_STRINGIFY` is True.
        """
        def wrapper(self, *args, **kwargs):
            res = func(self, *args, **kwargs)
            if isinstance(res, pd.DataFrame):
                return self._stringify(res)
            assert isinstance(res, tuple)
            return tuple(self._stringify(df) for df in res)
        return wrapper

    # ===================================================================
    #                         Public orchestrator
    # ===================================================================
    def load(self, regenerate: bool = False):
        """
        Main entry-point called by external scripts.

        If datasets already exist on disk (and regenerate=False) they are
        loaded; otherwise we re-generate everything from scratch.
        """
        pnt(f"load {self.get_name()} processor")

        # --------------------------------------------------------------
        # Fast path: datasets already serialised → just load them
        # --------------------------------------------------------------
        if (
            not regenerate
            and os.path.exists(self.item_save_dir)
            and os.path.exists(self.user_save_dir)
            and all(os.path.exists(self.get_save_dir(m)) for m in Interactions.modes)
        ):
            self.item = UniTok.load(self.item_save_dir)
            pnt(f"loaded {len(self.item)} items")

            self.user = UniTok.load(self.user_save_dir)
            pnt(f"loaded {len(self.user)} users")

            uts = [UniTok.load(self.get_save_dir(m)) for m in Interactions.modes]
            for m, ut in zip(Interactions.modes, uts):
                pnt(f"loaded {len(ut)} {m.name} interactions")
            self.interactions = Interactions(*uts)
            return  # early-exit

        # --------------------------------------------------------------
        # Slow path: raw → processed → serialised
        # --------------------------------------------------------------
        self.item_df = self.load_items()
        pnt(f"loaded {len(self.item_df)} items")

        self.user_df = self.load_users()
        pnt(f"loaded {len(self.user_df)} users")

        self.interactions = self.load_interactions()
        for m in Interactions.modes:
            pnt(f"loaded {len(self.interactions[m])} {m.name} interactions")

        # ---------- augment users with negative samples ---------------
        if self.NEG_COL:
            df = pd.concat([self.interactions.train_df, self.interactions.valid_df])
            neg_df = df[df[self.LBL_COL] == 0]
            neg_df = neg_df.groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()
            neg_df.columns = [self.UID_COL, self.NEG_COL]
            self.user_df = self.user_df.merge(neg_df, on=self.UID_COL, how="left")
            self.user_df[self.NEG_COL] = self.user_df[self.NEG_COL].apply(
                lambda x: x if isinstance(x, list) else []
            )

        # Proceed with vocab building / tokenization
        self.generate()

    # ===================================================================
    #                    Internal: tokenization pipeline
    # ===================================================================
    def generate(self):
        """
        1. Build vocabularies, trim unused entries.
        2. Compress item & user DataFrames.
        3. tokenize items, users and interactions.
        4. Persist everything to disk.
        """

        # ------------------------ USER VOCAB ---------------------------
        user_vocab = Vocab(name=self.UID_COL)
        user_vocab.extend(list(self.user_df[self.UID_COL]))
        user_vocab.counter.activate()
        for df in self.interactions.values():
            user_vocab.extend(list(df[self.UID_COL]))
        
        used_users = set(user_vocab[u] for u in user_vocab.counter.trim(min_count=1))
        self.user_df = self.user_df[self.user_df[self.UID_COL].isin(used_users)].reset_index(drop=True)
        pnt(f"compressed to {len(self.user_df)} users")

        # ------------------------ ITEM VOCAB ---------------------------
        slicer = Feature.get_slice(self.NEG_TRUNCATE)

        item_vocab = Vocab(name=self.IID_COL)
        item_vocab.extend(list(self.item_df[self.IID_COL]))
        item_vocab.counter.initialize()
        item_vocab.counter.activate()
        for df in self.interactions.values():
            item_vocab.extend(list(df[self.IID_COL]))
        self.user_df[self.HIS_COL].map(item_vocab.extend)
        self.user_df[self.NEG_COL].map(lambda h: item_vocab.extend(h[slicer]))

        used_items = set(item_vocab[i] for i in item_vocab.counter.trim(min_count=1))
        self.item_df = self.item_df[self.item_df[self.IID_COL].isin(used_items)].reset_index(drop=True)
        pnt(f"compressed to {len(self.item_df)} items")

        # --------------------- save raw parquet ------------------------
        self.item_df.to_parquet(os.path.join(self.save_dir, "items.parquet"))
        self.user_df.to_parquet(os.path.join(self.save_dir, "users.parquet"))
        self.interactions.train_df.to_parquet(os.path.join(self.save_dir, "train.parquet"))
        self.interactions.valid_df.to_parquet(os.path.join(self.save_dir, "valid.parquet"))
        self.interactions.test_df.to_parquet(os.path.join(self.save_dir, "test.parquet"))

        # ---------------------- tokenize ITEMS -------------------------
        with UniTok() as self.item:
            item_vocab = Vocab(name=self.IID_COL)
            self.item.add_feature(
                tokenizer=EntityTokenizer(vocab=item_vocab),
                column=self.IID_COL,
                name=self.IID_FEAT,
                key=True,
            )
            self.config_item_tokenization()
            self.item.tokenize(self.item_df).save(self.item_save_dir)
            pnt(f"tokenized {len(self.item)} items")

        # ---------------------- tokenize USERS -------------------------
        with UniTok() as self.user:
            VocabularyHub.add(item_vocab)

            user_vocab = Vocab(name=self.UID_COL)
            self.user.add_feature(
                tokenizer=EntityTokenizer(vocab=user_vocab),
                column=self.UID_COL,
                name=self.UID_FEAT,
                key=True,
            )
            self.user.add_feature(
                tokenizer=EntitiesTokenizer(vocab=item_vocab),
                column=self.HIS_COL,
                name=self.HIS_FEAT,
                truncate=0,
            )
            if self.NEG_COL:
                self.user.add_feature(
                    tokenizer=EntitiesTokenizer(vocab=self.IID_COL),
                    column=self.NEG_COL,
                    truncate=self.NEG_TRUNCATE,
                )
            self.config_user_tokenization()
            self.user.tokenize(self.user_df).save(self.user_save_dir)
            pnt(f"tokenized {len(self.user)} users")

        # -------------------- tokenize INTERACTIONS --------------------
        uts = []
        for mode in Interactions.modes:
            with UniTok() as ut:
                ut.add_index_feature()
                ut.add_feature(
                    tokenizer=EntityTokenizer(vocab=user_vocab),
                    column=self.UID_COL,
                    name=self.UID_FEAT,
                )
                ut.add_feature(
                    tokenizer=EntityTokenizer(vocab=item_vocab),
                    column=self.IID_COL,
                    name=self.IID_FEAT,
                )
                ut.add_feature(
                    tokenizer=DigitTokenizer(vocab=self.LBL_COL, vocab_size=2),
                    column=self.LBL_COL,
                    name=self.LBL_FEAT,
                )
                self.config_inter_tokenization(ut)
                ut.tokenize(self.interactions[mode]).save(self.get_save_dir(mode))
                pnt(f"tokenized {len(ut)} {mode.name} interactions")
            uts.append(ut)

        # Replace raw DataFrames with tokenized UniTok container
        self.interactions = Interactions(*uts)
