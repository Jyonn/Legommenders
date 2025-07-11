"""
base_pager.py

Generic **“batch-by-pagination”** helper.

The class is designed for scenarios where we iterate over an *arbitrary*
Python list (`self.contents`) but still want to process the data in
mini-batches – similar to what PyTorch’s `DataLoader` would do for
tensor datasets.

Workflow
--------
1)  The user provides
        • `contents`  – list[Any]       (data items to iterate over)
        • `model`     – Callable        (function / nn.Module that accepts
                                         **keyword** arguments that are
                                         produced by `get_features`)
        • `page_size` – int             (mini-batch size)

2)  `BasePager.run` loops through the list, calls
        self.get_features(content, index)
    for each item and accumulates the returned feature tensors until the
    desired batch size is reached.

3)  When a batch is full (or the input is exhausted) `_process()` is
    executed:
        • stack the features into tensors on `Env.device`,
        • perform a forward pass through `model`,
        • let the subclass-provided `combine()` decide what to do with
          the model output (store, aggregate, write to disk, …).

Sub-classes have to implement
-----------------------------
• `get_features(content, index)` -> dict[str, torch.Tensor]
    Extract per-item tensors that will be stacked later on.
• `combine(slices, features, output)` -> None
    Handle the model output.  `slices` indicates the *global* index range
    that the current batch corresponds to (useful for caching back into
    an array of the original order).

Note
----
This class *does not* impose any structure on the model’s signature
other than accepting keyword arguments.  If the model is an `nn.Module`
the caller is responsible for putting it into `.eval()` mode etc.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Any

import torch

from loader.env import Env
from utils import bars


class BasePager:
    """
    Mini-batch helper that paginates over a plain Python list.

    Parameters
    ----------
    contents : list
        Sequence of objects that will be fed into `get_features`.
    model : Callable
        Anything callable that consumes the *stacked* feature tensors
        (`model(**features)`).
    page_size : int
        Number of samples per mini-batch.
    desc : str, optional
        Description shown by the progress bar.
    **kwargs
        Ignored – placeholder so sub-classes can accept additional args.
    """

    def __init__(
        self,
        contents: List[Any],
        model: Callable,
        page_size: int,
        desc: str | None = None,
        **kwargs,
    ) -> None:
        self.contents = contents
        self.model = model
        self.page_size = page_size

        # runtime buffers ------------------------------------------------
        self.current: Dict[str, List[torch.Tensor]] = {}  # per-feature list
        self.current_count: int = 0                       # #items in buffer
        self.cache_count: int = 0                         # #batches processed

        self.desc = desc or "Pager Caching"

    # ------------------------------------------------------------------ #
    # Methods that **must** be overridden by sub-classes                 #
    # ------------------------------------------------------------------ #
    def get_features(self, content: Any, index: int) -> Dict[str, torch.Tensor]:
        """
        Convert a single `content` item into a dict of tensors.
        Must be implemented by sub-classes.
        """
        raise NotImplementedError

    def combine(
        self,
        slices: slice,
        features: Dict[str, torch.Tensor],
        output: Any,
    ) -> None:
        """
        Handle the model output of the current batch.
        Must be implemented by sub-classes.

        Parameters
        ----------
        slices : slice
            Global index range that this batch covers inside `contents`.
        features : dict[str, torch.Tensor]
            Mini-batch that was fed into the model (after stacking).
        output : Any
            Whatever the model returned.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Public control-flow                                                #
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """
        Iterate over `self.contents`, build mini-batches and forward them
        to `self.model`.  Progress is displayed via `bars.DescBar`.
        """
        for index, content in enumerate(bars.DescBar(desc=self.desc)(self.contents)):
            # 1) Extract per-item features ------------------------------
            features = self.get_features(content, index=index)

            # 2) Append to current buffer -------------------------------
            for feature, value in features.items():
                self.current.setdefault(feature, []).append(value)

            self.current_count += 1

            # 3) Process full batch -------------------------------------
            if self.current_count == self.page_size:
                self._process()

        # Handle remaining samples (if any)
        if self.current_count:
            self._process()

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def stack_features(self) -> Dict[str, torch.Tensor]:
        """
        Stack lists of tensors along dimension 0 and move them to the
        configured device.
        """
        return {
            feature: torch.stack(self.current[feature]).to(Env.device)
            for feature in self.current
        }

    def _process(self) -> None:
        """
        Internal helper that turns the *currently buffered* samples into
        a batch, executes the model and calls `combine`.
        """
        # Prepare mini-batch tensors ------------------------------------
        features = self.stack_features()

        # Forward pass --------------------------------------------------
        output = self.model(**features)

        # Global slice that corresponds to this batch
        start = self.cache_count * self.page_size
        end = start + self.current_count
        slices = slice(start, end)

        # Delegate result handling to sub-class
        self.combine(slices, features, output)

        # Reset buffers -------------------------------------------------
        self.cache_count += 1
        self.current_count = 0
        for feature in self.current:
            self.current[feature] = []
