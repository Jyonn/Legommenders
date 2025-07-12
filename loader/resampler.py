"""
resampler.py

Utility that *re-builds* / *augments* the raw samples coming from the
data-pipeline so that they can be directly fed into the Legommender
model.  
Tasks covered by the class:

1)  Candidate list preparation
    • negative sampling (during training / simple evaluation)
    • optional injection of item-level **content features**
      (embeddings, LM tokens …) – either taken from an *item cache*,
      or looked up on-the-fly
2)  Click-history preparation
    • padding / mask creation
    • optional injection of item content features for every clicked item
      (again, taken from cache when available)
3)  Efficient re-use of already computed representations
    • the class respects several global “caching flags” stored in
      `loader.env.Env` and skips redundant computations accordingly
4)  Stateless helper interface:  `resampler(sample)` returns the mutated
    sample dict, ready for collation.

The implementation is *heavily* driven by configuration flags because
there are many combinations of features / caches that can be enabled /
disabled.
"""

from __future__ import annotations
import random
from typing import List, Dict, Any

import torch

from loader.env import Env
from loader.data_set import DataSet
from model.lego_config import LegoConfig
from utils import bars
from utils.stacker import FastStacker
from utils.timer import Timer


class Resampler:
    """
    Pre-processing helper that converts a raw Python dict `sample`
    obtained from a `DataLoader` into a tensorized representation that
    fulfils the input contract of *Legommender*.

    Parameters
    ----------
    lego_config : LegoConfig
        Full model / training configuration which provides access to
        feature columns, operators, caching options, etc.
    """

    # ------------------------------------------------------------------ #
    # Constructor                                                        #
    # ------------------------------------------------------------------ #
    def __init__(self, lego_config: LegoConfig):
        # Helper for measuring execution time of costly paths
        self.timer = Timer(activate=True)

        # Store configuration and a few frequently used shortcuts
        self.lego_config = lego_config
        self.use_item_content = lego_config.use_item_content

        self.cm = lego_config.cm
        self.history_col = self.cm.history_col
        self.item_col = self.cm.item_col
        self.user_col = self.cm.user_col
        self.neg_col = self.cm.neg_col
        self.mask_col = self.cm.mask_col

        # ------------------------------------------------------------------
        # Item caching / content handling
        # ------------------------------------------------------------------
        self.item_dataset = None          # DataSet with *all* items
        self.item_inputer = None          # callable: sample -> content tensors
        self.item_cache: List[Dict[str, Any]] | None = None  # cached item tensors
        self.stacker = FastStacker(aggregator=torch.stack)

        if self.use_item_content:
            # Build a local cache that holds the *tensorized* content
            # representation for every single item.
            self.item_dataset = DataSet(ut=lego_config.item_ut)
            self.item_inputer = lego_config.item_operator.inputer
            self.item_cache = self._build_item_cache()

        # ------------------------------------------------------------------
        # User-level cache for click history tensors
        # ------------------------------------------------------------------
        self.user_cache: Dict[int, Dict[str, torch.Tensor]] = {}

        # ------------------------------------------------------------------
        # Click-history settings
        # ------------------------------------------------------------------
        self.user_inputer = lego_config.user_operator.inputer
        self.max_click_num: int = (
            self.user_inputer.ut.meta.features[self.history_col].max_len
        )

        # ------------------------------------------------------------------
        # Negative sampling
        # ------------------------------------------------------------------
        self.use_neg_sampling = lego_config.use_neg_sampling
        self.item_size: int = (
            lego_config.item_ut.meta.features[self.item_col].tokenizer.vocab.size
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _build_item_cache(self) -> list:
        """
        Pre-compute the *content embeddings* for **all** items once.

        Returns
        -------
        list[dict[str, torch.Tensor]]
            List indexed by item-id containing the nested tensor
            structure produced by `self.item_inputer`.
        """
        item_cache = []
        for sample in (bars.DescBar(desc="Building Item Cache"))(self.item_dataset):
            item_cache.append(self.item_inputer(sample))
        return item_cache

    # Static wrapper that keeps the dtype consistent
    @staticmethod
    def pack_tensor(array):
        """
        Convert a Python list / sequence into a 1-D `LongTensor`.
        """
        return torch.tensor(array, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # Candidate items                                                    #
    # ------------------------------------------------------------------ #
    def rebuild_candidates(self, sample: dict) -> None:
        """
        Prepare the candidate item list contained in `sample`.

        Steps
        -----
        1) Ensure that the field is a *list* (even for single item id).
        2) Perform negative sampling during training / simple dev eval.
        3) Decide whether content tensors have to be injected:
           • skip when   Env.lm_cache  or  Env.item_cache  is set
           • otherwise   stack the item tensors via `self.stacker`
        4) Replace the raw ids with one of:
             – `LongTensor(ids)`                             (no content)
             – nested structure of tensors                   (with content)
        """
        # Ensure we always operate on a *list* of ids
        if not isinstance(sample[self.item_col], list):
            sample[self.item_col] = [sample[self.item_col]]

        # ---------------------- negative sampling ---------------------- #
        if self.use_neg_sampling:
            if Env.is_training or (Env.is_evaluating and Env.simple_dev):
                # cross-entropy training → need explicit negatives
                true_negs = sample[self.neg_col] if self.neg_col else []
                rand_count = max(self.lego_config.neg_count - len(true_negs), 0)

                neg_samples = random.sample(
                    true_negs, k=min(self.lego_config.neg_count, len(true_negs))
                )
                neg_samples += [
                    random.randint(0, self.item_size - 1) for _ in range(rand_count)
                ]
                sample[self.item_col].extend(neg_samples)

        # Remove the negative column since it is now merged
        if self.neg_col:
            sample.pop(self.neg_col, None)

        # ------------------------------------------------------------------
        # Decide whether we can keep only the *ids* or have to attach
        # the *content tensors*.
        # ------------------------------------------------------------------
        if (
            not self.use_item_content
            or Env.lm_cache
            or Env.item_cache
        ):
            # Only ids are needed → convert into tensor and quit
            sample[self.item_col] = self.pack_tensor(sample[self.item_col])
            return

        # Inject *content tensors* (list-of-dict → batched dict of tensors)
        sample[self.item_col] = self.stacker(
            [self.item_cache[nid] for nid in sample[self.item_col]]
        )

    # ------------------------------------------------------------------ #
    # Click history                                                      #
    # ------------------------------------------------------------------ #
    def rebuild_clicks(self, sample: dict) -> None:
        """
        Prepare the click history contained in `sample`.

        Handles padding / attention mask creation and optional
        content-tensor retrieval similar to `rebuild_candidates`.
        """
        if Env.user_cache:
            # Full user representation already cached → remove history
            sample.pop(self.history_col, None)
            return

        # -------------------------------------------------------------- #
        # Convert raw list of item ids into padded tensor representation #
        # -------------------------------------------------------------- #
        len_clicks = len(sample[self.history_col])

        # Build the attention mask (1=real click, 0=padding)
        sample[self.mask_col] = [1] * len_clicks + [0] * (
            self.max_click_num - len_clicks
        )
        sample[self.mask_col] = torch.tensor(sample[self.mask_col], dtype=torch.long)

        # Pad the id list when item content will be accessed later
        if self.use_item_content:
            sample[self.history_col].extend([0] * (self.max_click_num - len_clicks))

        # ------------------------------------------------------------------
        # Case A – no item content involved → let the *user inputer* handle
        #          the full transformation (might flatten / embed etc.).
        # ------------------------------------------------------------------
        if not self.use_item_content:
            sample[self.history_col] = self.user_inputer(sample)
            return

        # Special case: flatten mode prepares its own masks
        if self.lego_config.user_operator_class.flatten_mode:
            sample[self.history_col] = self.user_inputer(sample)
            sample[self.mask_col] = self.user_inputer.get_mask(sample[self.history_col])
            return

        # ------------------------------------------------------------------
        # At this point we know that item content *is* used…
        # …but we might still be able to skip heavy tensor look-ups when
        # the global caches are in place.
        # ------------------------------------------------------------------
        if Env.lm_cache or Env.item_cache:
            sample[self.history_col] = self.pack_tensor(sample[self.history_col])
            return

        # ------------------------------------------------------------------
        # Retrieve / cache the content tensors for the clicked items
        # ------------------------------------------------------------------
        if sample[self.user_col] in self.user_cache:
            # Fast path: we have seen this user before
            sample[self.history_col] = self.user_cache[sample[self.user_col]]
        else:
            # Build the tensor batch and store it for later re-use
            sample[self.history_col] = self.stacker(
                [self.item_cache[nid] for nid in sample[self.history_col]]
            )
            self.user_cache[sample[self.user_col]] = sample[self.history_col]

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #
    def rebuild(self, sample: dict) -> dict:
        """
        Entry point that mutates `sample` *in-place* and returns it.
        """
        self.rebuild_candidates(sample)
        self.rebuild_clicks(sample)
        return sample

    # Syntactic sugar → allows `resampler(sample)`
    def __call__(self, sample: dict) -> dict:
        return self.rebuild(sample)
