"""
mind_processor.py

A concrete implementation of `BaseProcessor` for the Microsoft News
Dataset (MIND).  The class handles:

    • Loading / cleaning raw TSV files from the official MIND release.
    • Building item / user / interaction DataFrames.
    • Constructing multiple text tokenizers (BERT, LLaMA-1, GloVe).
    • Sampling a 10 % user split for validation, rest for training.
    • Injecting negative samples for recommendation algorithms.

The original logic from the prompt is **left untouched**; only clarifying
comments have been added.
"""

import os
import random
from typing import cast

import pandas as pd
from unitok import BertTokenizer, TransformersTokenizer, EntityTokenizer
from unitok.tokenizer.glove_tokenizer import GloVeTokenizer

from embedder.glove_embedder import GloVeEmbedder
from processor.base_processor import BaseProcessor, Interactions
from utils.config_init import ModelInit


class MINDProcessor(BaseProcessor):
    # ------------------------------------------------------------------ #
    # Column names in raw files
    # ------------------------------------------------------------------ #
    IID_COL = "nid"       # unique news id
    UID_COL = "uid"       # unique user id
    HIS_COL = "history"   # clicked history (space-separated nids)
    LBL_COL = "click"     # click label (0/1)
    NEG_COL = "neg"       # negative samples column (filled later)

    # ------------------------------------------------------------------ #
    # Behaviour flags
    # ------------------------------------------------------------------ #
    NEG_TRUNCATE = 100        # at most 100 negatives kept per user
    REQUIRE_STRINGIFY = False # nids/uids are already strings in MIND

    # ------------------------------------------------------------------ #
    # Meta information about textual attributes of a news item
    # attr_name -> max_token_length  (0 means “no truncation”)
    # ------------------------------------------------------------------ #
    @property
    def attrs(self) -> dict:
        return dict(
            title=50,
            abstract=200,
            category=0,
            subcategory=0,
        )

    # ================================================================== #
    #               Tokeniser configuration for ITEM UniTok
    # ================================================================== #
    def config_item_tokenization(self):
        """
        Add several language models’ tokenizers for rich text encoding,
        plus categorical tokenizers for (sub)category fields.
        """
        # Pre-trained tokenizers ------------------------------------------------
        bert_tokenizer = BertTokenizer(vocab="bert")
        llama1_tokenizer = TransformersTokenizer(
            vocab="llama1",
            key=ModelInit.get("llama1"),  # path or HuggingFace name
        )
        glove_tokenizer = GloVeTokenizer(vocab=GloVeEmbedder.get_glove_vocab())

        # Register tokenizers for every textual attribute ----------------------
        self.add_item_tokenizer(bert_tokenizer)
        self.add_item_tokenizer(llama1_tokenizer)
        self.add_item_tokenizer(glove_tokenizer)

        # Simple categorical tokenizers ----------------------------------------
        self.item.add_feature(
            tokenizer=EntityTokenizer(vocab="category"),
            column="category",
        )
        self.item.add_feature(
            tokenizer=EntityTokenizer(vocab="subcategory"),
            column="subcategory",
        )

    # ================================================================== #
    #                         Raw file loaders
    # ================================================================== #
    def _load_items(self, path: str) -> pd.DataFrame:
        """
        Load a `news.tsv` file and keep only the relevant columns.
        """
        return pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep="\t",
            names=[
                self.IID_COL, "category", "subcategory",
                "title", "abstract", "url", "tit_ent", "abs_ent",
            ],
            usecols=[self.IID_COL, "category", "subcategory", "title", "abstract"],
        )

    def load_items(self) -> pd.DataFrame:
        """
        Merge *train* and *dev* splits to obtain the complete catalogue of
        news articles; add prompt prefixes to facilitate LLM inputs.
        """
        train_df = self._load_items(os.path.join(self.data_dir, "train", "news.tsv"))
        valid_df = self._load_items(os.path.join(self.data_dir, "dev", "news.tsv"))
        item_df = pd.concat([train_df, valid_df]).drop_duplicates([self.IID_COL])

        # Clean-ups & prompt columns ------------------------------------------
        item_df["abstract"] = item_df["abstract"].fillna("")
        item_df["prompt"]              = "Here is a piece of news article. "
        item_df["prompt_title"]        = "Title: "
        item_df["prompt_abstract"]     = "Abstract: "
        item_df["prompt_category"]     = "Category: "
        item_df["prompt_subcategory"]  = "Subcategory: "
        return item_df

    # ------------------------------------------------------------------ #
    def _load_users(self, path: str) -> pd.DataFrame:
        """
        Load a `behaviors.tsv` file keeping uid + history only.
        """
        return pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep="\t",
            names=["imp", self.UID_COL, "time", self.HIS_COL, "predict"],
            usecols=[self.UID_COL, self.HIS_COL],
        )

    def load_users(self) -> pd.DataFrame:
        """
        Combine train & dev users, clean click histories so only news that
        **exist in item_df** remain; drop users with empty histories.
        """
        item_set = set(self.item_df[self.IID_COL].unique())

        train_df = self._load_users(os.path.join(self.data_dir, "train", "behaviors.tsv"))
        valid_df = self._load_users(os.path.join(self.data_dir, "dev", "behaviors.tsv"))
        users = pd.concat([train_df, valid_df]).drop_duplicates([self.UID_COL])

        # Convert “nid1 nid2 …” → list[str]
        users[self.HIS_COL] = users[self.HIS_COL].str.split()
        users = users.dropna(subset=[self.HIS_COL])

        # Keep only valid nids present in the catalogue
        users[self.HIS_COL] = users[self.HIS_COL].apply(
            lambda lst: [nid for nid in lst if nid in item_set]
        )
        users = users[users[self.HIS_COL].map(lambda lst: len(lst) > 0)]
        return users

    # ------------------------------------------------------------------ #
    def _load_interactions(self, path: str) -> pd.DataFrame:
        """
        Parse behaviour logs into (uid, nid, click) tuples:
            predict column is "nid1-0 nid2-1 …" → explode rows.
        """
        user_set = set(self.user_df[self.UID_COL].unique())

        interactions = pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep="\t",
            names=["imp", self.UID_COL, "time", self.HIS_COL, "predict"],
            usecols=[self.UID_COL, "predict"],
        )
        interactions = interactions[interactions[self.UID_COL].isin(user_set)]

        # Split “nid-click” tokens and explode rows ---------------------------
        interactions["predict"] = interactions["predict"].str.split().apply(
            lambda x: [token.split("-") for token in x]
        )
        interactions = interactions.explode("predict")
        interactions[[self.IID_COL, self.LBL_COL]] = pd.DataFrame(
            interactions["predict"].tolist(), index=interactions.index
        )
        interactions.drop(columns=["predict"], inplace=True)
        interactions[self.LBL_COL] = interactions[self.LBL_COL].astype(int)
        return interactions

    def load_interactions(self) -> Interactions:
        """
        Build TRAIN / VALID (10 % users) / TEST interaction sets.
        """
        train_df = self._load_interactions(os.path.join(self.data_dir, "train", "behaviors.tsv"))
        test_df  = self._load_interactions(os.path.join(self.data_dir, "dev", "behaviors.tsv"))

        # --------------- stratified user-split for validation ---------
        users = list(train_df[self.UID_COL].unique())
        random.shuffle(users)
        valid_users = set(users[: int(len(users) * 0.1)])

        valid_df = train_df[train_df[self.UID_COL].isin(valid_users)]
        train_df = train_df[~train_df[self.UID_COL].isin(valid_users)]

        # Re-index for cleanliness
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)

        return Interactions(train_df, valid_df, test_df)

    # ================================================================== #
    #     Hook executed by BaseProcessor.generate() before tokenization
    # ================================================================== #
    def after_load_interactions(self):
        """
        Compute per-user negative samples from TRAIN+VALID splits and
        merge them into `self.user_df` under column NEG_COL.
        """
        df = pd.concat([self.interactions.train_df, self.interactions.valid_df])
        neg_df = df[df[self.LBL_COL] == 0]
        # Aggregate list of negatives per user
        neg_df = neg_df.groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()
        neg_df.columns = [self.UID_COL, self.NEG_COL]

        # Left-join onto user table; NaN → []
        self.user_df = self.user_df.merge(neg_df, on=self.UID_COL, how="left")
        self.user_df[self.NEG_COL] = self.user_df[self.NEG_COL].apply(
            lambda x: x if isinstance(x, list) else []
        )
