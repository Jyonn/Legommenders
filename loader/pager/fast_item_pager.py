"""
fast_item_pager.py

Sub-class of `BasePager` that builds / caches *item* representations
("fast item repr") in mini-batches.

Typical workflow
----------------
1)  A list of *items* (e.g. product ids) is handed over via `contents`
    when the pager is instantiated (see super-class signature).
2)  During `run()` the pager
        • extracts the required tensors per item (`get_features`)
        • feeds them into the provided `model`
        • writes the resulting hidden state into the pre-allocated
          tensor `self.fast_item_repr`.

Special behaviour controlled by `Env.lm_cache`
----------------------------------------------
• When `Env.lm_cache == True` the language-model embedding lookup is
  performed elsewhere and therefore **only the item index** is needed as
  input for the downstream model.  In that mode  
      get_features → {"embeddings": tensor(index), "mask": None}

• Otherwise the pager leverages a user-supplied `inputer` (implements
  `BaseInputer`) to obtain
      embeddings : tensor(batch, seq_len, dim)
      mask       : tensor(batch, seq_len)
  for every item.

The class further deals with nested dictionaries returned by some
inputers (e.g. hierarchical tokenisers) via `utils.stacker.Stacker`.
"""

from __future__ import annotations

import torch

from loader.env import Env
from model.inputer.base_inputer import BaseInputer
from utils.stacker import Stacker
from loader.pager.base_pager import BasePager


class FastItemPager(BasePager):
    """
    Pager that populates `self.fast_item_repr` with the model’s output
    batch-by-batch.

    Parameters
    ----------
    inputer : BaseInputer
        Helper that transforms a *single* item into tensors
        (embeddings / masks) when `Env.lm_cache` is *False*.
    hidden_size : int
        Dimensionality of the model’s output (only used for assertions
        outside this snippet).
    placeholder : torch.Tensor
        Pre-allocated tensor whose 1st dimension equals `len(contents)`.
        The pager writes the model output directly into this buffer.
    **kwargs
        Forwarded to `BasePager`, must include at least
            contents : list
            model    : Callable
            page_size: int
    """

    def __init__(
        self,
        inputer: BaseInputer,
        hidden_size: int,
        placeholder: torch.Tensor,
        **kwargs,
    ):
        # parent handles: contents / model / page_size (+ progress bar)
        super().__init__(desc="Item Caching", **kwargs)

        self.inputer = inputer
        self.hidden_size = hidden_size
        # Destination for the computed item representations
        self.fast_item_repr = placeholder.to(Env.device)

        # Helper that can stack nested dict structures
        self.stacker = Stacker(aggregator=torch.stack)

    # ------------------------------------------------------------------ #
    # BasePager overrides                                                #
    # ------------------------------------------------------------------ #
    def get_features(self, content, index: int) -> dict:
        """
        Convert a single *content* element into model input tensors.
        """
        if Env.lm_cache:
            # Upstream caching already mapped items to embedding rows
            # -> model expects just an index placeholder + dummy mask
            return dict(
                embeddings=torch.tensor(index),
                mask=None,
            )

        # Otherwise perform the full embedding lookup via `inputer`
        return dict(
            embeddings=self.inputer.get_embeddings(content),
            mask=self.inputer.get_mask(content),
        )

    def stack_features(self) -> dict:
        """
        Stack the per-item features accumulated in `self.current`.

        Takes care of two cases:
        • plain tensors    -> `torch.stack`
        • nested dicts     -> `utils.stacker.Stacker` (recursive stack)
        """
        features = {}
        # If lm_cache is enabled we only feed "embeddings" (plus dummy mask)
        feature_cols = ["embeddings"] if Env.lm_cache else self.current

        if Env.lm_cache:
            features["mask"] = None  # keep signature consistent

        for feature in feature_cols:
            first_entry = self.current[feature][0]
            if isinstance(first_entry, torch.Tensor):
                # Simple tensor list
                features[feature] = torch.stack(self.current[feature]).to(Env.device)
            else:
                # Nested structures (dict[str, tensor])
                assert isinstance(first_entry, dict)
                features[feature] = self.stacker(
                    self.current[feature],
                    apply=lambda x: x.to(Env.device),
                )
        return features

    def combine(self, slices: slice, features: dict, output: torch.Tensor) -> None:
        """
        Write the model output of the current batch into
        `self.fast_item_repr` at the correct position.

        `slices` corresponds to the *global* range of items covered by
        the batch, therefore we can assign directly.
        """
        self.fast_item_repr[slices] = output.detach()
