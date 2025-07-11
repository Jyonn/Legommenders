"""
legommender.py

High-level *end-to-end* model that brings together

    • an (optional) *item* operator          – encodes every catalogue
                                               entry into a fixed vector
    • a            *user* operator          – encodes the click history
    • a            *predictor*              – produces a match / rank
                                               score for (user, item)
    • several utility helpers               – caching, shaper …

The surrounding infrastructure (datasets, column map, embedding hub,
negative sampling, …) is defined in `LegoConfig` which is passed to the
constructor and kept as `self.config`.

`Legommender` offers a *single* public interface:
    >>> loss_or_scores = model(batch)

For **training / (simple) dev-evaluation** it returns the loss
(Cross-Entropy or BCE, depending on the sampling mode), while for
**testing** or *full* evaluation it directly returns the raw *scores*
for further metric computation.

Key features
------------
1)  Supports both
        • matching with negative samples  (scores shape: B × (K+1))
        • ranking  of a single candidate (scores shape: B × 1)
2)  Transparent representation caching:
        • `ItemCacher` / `UserCacher` accelerate repeated evaluation
          by skipping redundant forward passes.
3)  Can fall back to *language-model caches* (LLMOperator) when present.
4)  Clean separation between   *data pre-processing* (`Resampler`) and
    *model forward pass* (`Legommender`).

"""

from __future__ import annotations

from typing import Tuple, List

import torch
from torch import nn
from pigmento import pnt  # coloured console printing

from loader.env import Env
from model.lego_config import LegoConfig
from loader.cacher.repr_cacher import ReprCacher
from loader.column_map import ColumnMap
from model.operators.lm_operator import LMOperator
from utils.shaper import Shaper


class Legommender(nn.Module):
    """
    End-to-end recommender model.

    Parameters
    ----------
    config : LegoConfig
        Fully built configuration (operators, predictor, column map …).
        Must call `config.build_components()` and
        `config.register_inputer_vocabs()` **before** instantiating this
        class.
    """

    # ------------------------------------------------------------------ #
    # Constructor                                                        #
    # ------------------------------------------------------------------ #
    def __init__(self, config: LegoConfig):
        super().__init__()

        # Store reference for later use
        self.config: LegoConfig = config

        # ------ flags & shortcuts ------------------------------------ #
        self.user_operator_class = config.user_operator_class
        self.predictor_class = config.predictor_class

        self.use_neg_sampling = config.use_neg_sampling
        self.neg_count = config.neg_count

        # Global embedding hub (for trainable token embeddings)
        self.eh = config.eh
        # Expose the embedding tables as *buffers* to make them visible
        # in `.named_parameters()` (and therefore in the printed summary)
        self.embedding_vocab_table = self.eh.vocab_table
        self.embedding_feature_table = self.eh.feature_table

        # Dataset specific information
        self.user_hub = config.user_ut
        self.item_hub = config.item_ut
        self.cm: ColumnMap = config.cm

        # ------ core components -------------------------------------- #
        self.flatten_mode = self.user_operator_class.flatten_mode
        self.item_op = config.item_operator          # may be None
        self.user_op = config.user_operator
        self.predictor = config.predictor

        # ------ utils ------------------------------------------------- #
        # Decide whether the LM operator employs an *internal* cache.
        Env.set_lm_cache(False)
        if config.use_item_content and isinstance(self.item_op, LMOperator):
            Env.set_lm_cache(self.item_op.use_lm_cache())
        pnt(f"set llm cache: {Env.lm_cache}")

        self.shaper = Shaper()          # reshaping helper for 3-D↔2-D
        self.cacher = ReprCacher(self)  # item / user representation cache
        self.cacher.activate(config.use_fast_eval)

        # Loss function depends on training objective
        self.loss_func = (
            nn.CrossEntropyLoss()
            if self.use_neg_sampling
            else nn.BCEWithLogitsLoss()
        )

    # ------------------------------------------------------------------ #
    # Helper – sample size determination                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_sample_size(item_content) -> int:
        """
        Determine the “batch size” of `item_content` irrespective of
        whether it is a plain tensor or a nested dict of tensors.
        """
        if isinstance(item_content, torch.Tensor):
            return item_content.shape[0]
        # Assume dict[str -> Tensor]
        first_key = next(iter(item_content.keys()))
        return item_content[first_key].shape[0]

    # ------------------------------------------------------------------ #
    # Item content                                                       #
    # ------------------------------------------------------------------ #
    def get_item_content(self, batch: dict, col: str):
        """
        Retrieve / compute the *content representation* for the
        candidate items or the click history.

        Returns a tensor with shape
            – (B , K , D)  when col == item_col    (candidates)
            – (B , S , D)  when col == history_col (clicks)
        where
            B – batch size,
            K – number of candidates,
            S – sequence length of clicks,
            D – hidden size
        """
        # -------------- 1) try *cached* representation -------------- #
        if self.cacher.item.cached:
            indices = batch[col]
            flat_indices = indices.reshape(-1)
            item_repr = self.cacher.item.repr[flat_indices]
            return item_repr.reshape(*indices.shape, -1)

        # -------------- 2) need to compute embeddings --------------- #
        if not Env.lm_cache:
            # Content comes from the operator’s inputer
            _orig_shape = None
            item_content = self.shaper.transform(batch[col])
            mask = self.item_op.inputer.get_mask(item_content)
            item_content = self.item_op.inputer.get_embeddings(item_content)
        else:
            # LM cache: `batch[col]` already contains token ids
            _orig_shape = batch[col].shape
            item_content = batch[col].reshape(-1)
            mask = None  # model will supply its own
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        # -------------- 3) batched forward through item_op ---------- #
        sample_size = self._get_sample_size(item_content)
        max_batch = self.config.item_page_size or sample_size
        num_batches = (sample_size + max_batch - 1) // max_batch

        # Pre-allocate → directly on proper device
        item_repr = self.item_op.get_full_placeholder(sample_size).to(Env.device)

        for i in range(num_batches):
            start, end = i * max_batch, min((i + 1) * max_batch, sample_size)
            sub_mask = None if mask is None else mask[start:end]
            item_repr[start:end] = self.item_op(item_content[start:end], mask=sub_mask)

        # Reshape back to original layout when LM cache was used
        if Env.lm_cache:
            item_repr = item_repr.view(*_orig_shape, -1)
        else:
            item_repr = self.shaper.recover(item_repr)

        return item_repr

    # ------------------------------------------------------------------ #
    # User content                                                       #
    # ------------------------------------------------------------------ #
    def get_user_content(self, batch: dict):
        """
        Retrieve / compute the user representation.
        """
        # Fast path – full user already cached
        if self.cacher.user.cached:
            return self.cacher.user.repr[batch[self.cm.user_col]]

        # Otherwise build it from click history
        if self.config.use_item_content and not self.flatten_mode:
            clicks = self.get_item_content(batch, self.cm.history_col)
        else:
            clicks = self.user_op.inputer.get_embeddings(batch[self.cm.history_col])

        return self.user_op(
            clicks,
            mask=batch[self.cm.mask_col].to(Env.device),
        )

    # ------------------------------------------------------------------ #
    # Forward                                                           #
    # ------------------------------------------------------------------ #
    def forward(self, batch: dict):
        """
        Main inference / training entry point.

        Behaviour depends on the global flags stored in `loader.env.Env`:

        Env.is_testing == True OR (Env.is_evaluating and not Env.simple_dev)
            → returns *scores*  (no loss)

        otherwise
            → returns *loss*
        """
        # Ensure candidate dimension is present (B × 1)
        if isinstance(batch[self.cm.item_col], torch.Tensor) and batch[self.cm.item_col].dim() == 1:
            batch[self.cm.item_col] = batch[self.cm.item_col].unsqueeze(1)

        # -------------- embeddings ---------------------------------- #
        # item side
        if self.config.use_item_content:
            item_embeddings = self.get_item_content(batch, self.cm.item_col)
        else:
            vocab_name = (
                self.config.user_ut.meta.features[self.cm.history_col]
                .tokenizer.vocab.name
            )
            item_embeddings = self.eh(vocab_name, col_name=self.cm.history_col)(
                batch[self.cm.item_col].to(Env.device)
            )

        # user side
        user_embeddings = self.get_user_content(batch)

        # -------------- score computation --------------------------- #
        if self.use_neg_sampling:
            scores = self._predict_for_neg_sampling(item_embeddings, user_embeddings)
            labels = torch.zeros(scores.size(0), dtype=torch.long, device=Env.device)
        else:
            scores = self._predict_for_ranking(item_embeddings, user_embeddings)
            labels = batch[self.cm.label_col].float().to(Env.device)

        # -------------- return depending on run-mode ---------------- #
        if Env.is_testing or (Env.is_evaluating and not Env.simple_dev):
            return scores

        return self.loss_func(scores, labels)

    # ------------------------------------------------------------------ #
    # Internal prediction helpers                                        #
    # ------------------------------------------------------------------ #
    def _predict_for_neg_sampling(self, item_embeddings, user_embeddings):
        """
        Compute the matching scores for (user, candidate*) pairs when
        *negative sampling* is enabled.
        """
        batch_size, candidate_size, hidden_size = item_embeddings.shape

        if self.predictor.keep_input_dim:
            return self.predictor(user_embeddings, item_embeddings)

        # Expand user → (B*K , D) , reshape items same way
        user_embeddings = self.user_op.prepare_for_predictor(user_embeddings, candidate_size)
        item_embeddings = item_embeddings.view(-1, hidden_size)

        scores = self.predictor(user_embeddings, item_embeddings)
        return scores.view(batch_size, -1)

    def _predict_for_ranking(self, item_embeddings, user_embeddings):
        """
        Compute the prediction scores in *ranking* setup
        (exactly one candidate per sample).
        """
        return self.predictor(user_embeddings, item_embeddings.squeeze(1))

    # ------------------------------------------------------------------ #
    # Convenience printing                                               #
    # ------------------------------------------------------------------ #
    def __str__(self):
        return self.__class__.__name__

    __repr__ = __str__

    # ------------------------------------------------------------------ #
    # Parameter grouping (pre-trained vs other)                          #
    # ------------------------------------------------------------------ #
    def get_parameters(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """
        Separate parameters into

            • `pretrained_parameters` – belonging to the *item operator*
              that should be fine-tuned with a lower learning rate
            • `other_parameters`      – everything else

        Returns
        -------
        tuple(list[nn.Parameter], list[nn.Parameter])
        """
        pretrained, other = [], []
        pretrained_names, other_names = [], []

        signals = self.item_op.get_pretrained_parameter_names()

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(name.startswith(f"item_op.{s}") for s in signals):
                pretrained.append(param)
                pretrained_names.append((name, tuple(param.shape)))
            else:
                other.append(param)
                other_names.append((name, tuple(param.shape)))

        # Pretty print for debugging
        for n, shp in pretrained_names:
            pnt(f"[P] {n} {shp}")
        for n, shp in other_names:
            pnt(f"[N] {n} {shp}")

        return pretrained, other
