"""
fast_user_pager.py

Sub-class of `BasePager` that builds / caches *user* representations
("fast user repr") in mini-batches.  It closely mirrors the behaviour of
`FastItemPager` but operates on (arbitrary) *user* features that are
handed over to the pager **already pre-computed** – therefore no
additional `inputer` is required.

Typical workflow
----------------
1) A list of *user feature dictionaries* is handed over via `contents`
   when the pager is instantiated (see super-class signature).

2) During `run()` the pager
       • copies the per-user feature dictionaries into an internal
         cache (`self.current`) while paging through `contents`
       • calls `stack_features()` to transform this list-of-dicts into
         proper batched tensors
       • feeds the batch into the provided `model`
       • writes the resulting hidden states into the pre-allocated tensor
         `self.fast_user_repr` (one row per user).

The class has no special handling for `Env.lm_cache`, because the user
features are assumed to be final tensors (or nested dictionaries of
tensors) already.

"""

from __future__ import annotations

import torch

from loader.env import Env
from utils.stacker import FastStacker
from loader.pager.base_pager import BasePager


class FastUserPager(BasePager):
    """
    Pager that populates `self.fast_user_repr` with the model’s output
    batch-by-batch.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the model’s output (only used for assertions
        outside this snippet).
    placeholder : torch.Tensor
        Pre-allocated tensor whose 1st dimension equals `len(contents)`.
        The pager writes the model output directly into this buffer.
    **kwargs
        Forwarded to `BasePager`, must include at least
            contents : list[dict | Any]
            model    : Callable
            page_size: int
    """

    # ------------------------------------------------------------------ #
    # Constructor                                                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        hidden_size: int,
        placeholder: torch.Tensor,
        **kwargs,
    ):
        # Let BasePager take care of tracking progress, slicing, etc.
        super().__init__(desc="User Caching", **kwargs)

        self.hidden_size = hidden_size

        # Destination for the computed user representations
        # (moved to the correct device right away).
        self.fast_user_repr = placeholder.to(Env.device)

        # Helper that can stack nested dict structures
        # (recursively applies torch.stack on matching keys).
        self.stacker = FastStacker(aggregator=torch.stack)

    # ------------------------------------------------------------------ #
    # BasePager overrides                                                #
    # ------------------------------------------------------------------ #
    def stack_features(self) -> dict:
        """
        Transform the list-of-features stored in `self.current`
        into proper batched tensors.

        The method is able to cope with three scenarios:
        • `Tensor` objects                -> `torch.stack`
        • nested dictionaries of tensors  -> `FastStacker` (recursive)
        • primitive python objects (ints) -> `torch.tensor`
        """
        stacked: dict[str, torch.Tensor | dict] = {}

        for feature in self.current:
            target = self.current[feature]

            # Case 1: simple tensor list
            if isinstance(target[0], torch.Tensor):
                stacked[feature] = torch.stack(target).to(Env.device)

            # Case 2: nested dict structures, e.g. {"a": tensor, "b": tensor}
            elif isinstance(target[0], dict):
                # FastStacker will move the result to the correct device
                stacked[feature] = self.stacker(target)

            # Case 3: numbers / strings that can be represented as tensors
            else:
                stacked[feature] = torch.tensor(target).to(Env.device)

        # The downstream model expects the whole bundle under key "batch"
        return dict(batch=stacked)

    def get_features(self, content, index: int) -> dict:
        """
        Return the raw *content* element itself.

        Contents have already been turned into feature dicts before they
        reach the pager, therefore no additional processing is required.
        """
        return content

    def combine(self, slices: slice, features: dict, output: torch.Tensor) -> None:
        """
        Write the model output of the current batch into
        `self.fast_user_repr` at the correct position.

        `slices` corresponds to the *global* range of users covered by
        the batch, therefore we can assign directly.
        """
        # `.detach()` keeps the cached representation out of the graph
        self.fast_user_repr[slices] = output.detach()
