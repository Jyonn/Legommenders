"""
oncemind_processor.py

A thin wrapper around `MINDProcessor` that supports the *ONCE-MIND*
benchmark in which the **DEV behaviours file** is split into two
non-overlapping subsets (`valid_df`, `test_df`) according to a list of
impression-ids (`imp_list`) provided by the author.

The path to the raw data is supplied via the special syntax

    "<data_root>$<imp_list.json>"

so that both directories can be passed as a single CLI argument.  Aside
from the extra `imp_id` feature and the custom split logic the remaining
behaviour is identical to the original `MINDProcessor`.
"""

import os
from typing import cast, List, Tuple

import pandas as pd
from unitok import JsonHandler, UniTok, EntityTokenizer

from processor.base_processor import Interactions
from processor.mind_processor import MINDProcessor


class ONCEMINDProcessor(MINDProcessor):
    """
    Extends `MINDProcessor` by:
        • adding an impression-id (IMP_COL) feature to the interaction
          UniTok,
        • allowing the dev set to be split deterministically according to
          a JSON list of impression ids.
    """

    IMP_COL = "imp"  # column name holding impression identifiers

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(self, data_dir: str | None = None):
        # Will hold the list of impression ids parsed from JSON
        self.imp_list: List[str] | None = None

        # Custom syntax: "<data_path>$<imp_json>"
        if data_dir and "$" in data_dir:
            data_dir, imp_path = data_dir.split("$")
            self.imp_list = JsonHandler.load(imp_path)
        elif data_dir:  # data_dir supplied but no '$' → raise error
            raise ValueError(
                "data_dir for ONCEMINDProcessor should use '$' to "
                "concatenate the data path and the imp path"
            )

        super().__init__(data_dir=data_dir)

    # ------------------------------------------------------------------ #
    # Extra feature for interaction UniTok                               #
    # ------------------------------------------------------------------ #
    def config_inter_tokenization(self, ut: UniTok):
        """
        Add impression-id as an Entity feature so that models can condition
        on the *context* (impression) if desired.
        """
        ut.add_feature(
            tokenizer=EntityTokenizer(vocab="imp_id"),
            column=self.IMP_COL,
        )

    # ------------------------------------------------------------------ #
    # Interaction CSV/TSV loader                                         #
    # (inherits user/item logic from parent class)                       #
    # ------------------------------------------------------------------ #
    def _load_interactions(self, path: str) -> pd.DataFrame:
        """
        Parse a behaviours.tsv file, exploding the ‘predict’ column to
        obtain one (uid, imp, nid, click) row per candidate news.
        """
        user_set = set(self.user_df[self.UID_COL].unique())

        interactions = pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep="\t",
            names=[self.IMP_COL, self.UID_COL, "time", self.HIS_COL, "predict"],
            usecols=[self.IMP_COL, self.UID_COL, "predict"],
        )

        # Keep only users we actually have
        interactions = interactions[interactions[self.UID_COL].isin(user_set)]

        # Split "nid-click" tokens
        interactions["predict"] = interactions["predict"].str.split().apply(
            lambda lst: [token.split("-") for token in lst]
        )
        interactions = interactions.explode("predict")
        interactions[[self.IID_COL, self.LBL_COL]] = pd.DataFrame(
            interactions["predict"].tolist(), index=interactions.index
        )
        interactions.drop(columns=["predict"], inplace=True)
        interactions[self.LBL_COL] = interactions[self.LBL_COL].astype(int)
        return interactions

    # ------------------------------------------------------------------ #
    # Helper: split impression list into N portions                      #
    # ------------------------------------------------------------------ #
    def splitter(self, portions: List[int]) -> List[List[str]]:
        """
        Divide `self.imp_list` into chunks proportional to *portions*.

        Example
        -------
        splitter([5, 5])  # → two equally sized halves
        """
        sum_portions = sum(portions)
        # Calculate absolute sizes for each split
        portions = [int(p * len(self.imp_list) / sum_portions) for p in portions]
        portions[-1] = len(self.imp_list) - sum(portions[:-1])  # remainder → last

        current_position = 0
        imp_lists = []
        for portion in portions:
            imp_lists.append(self.imp_list[current_position : current_position + portion])
            current_position += portion
        return imp_lists

    # ------------------------------------------------------------------ #
    # Public loader that creates train / valid / test DataFrames         #
    # ------------------------------------------------------------------ #
    def load_interactions(self) -> Interactions:
        """
        Split the *dev* behaviours into VALID and TEST according to the
        impression list read in `__init__`.
        """
        train_df = self._load_interactions(os.path.join(self.data_dir, "train", "behaviors.tsv"))
        dev_df   = self._load_interactions(os.path.join(self.data_dir, "dev", "behaviors.tsv"))

        # Divide impression ids 50/50 into dev_imps & test_imps ----------
        dev_imps, test_imps = self.splitter([5, 5])
        dev_list, test_list = [], []

        # Allocate each impression to its target split
        for imp, imp_df in dev_df.groupby(self.IMP_COL):
            (dev_list if imp in dev_imps else test_list).append(imp_df)

        valid_df = pd.concat(dev_list, ignore_index=True)
        test_df  = pd.concat(test_list, ignore_index=True)

        # Clean indices
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)

        return Interactions(train_df, valid_df, test_df)
