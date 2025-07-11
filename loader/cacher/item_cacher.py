"""
item_cacher.py

High-level wrapper that leverages `FastItemPager` to *pre-compute* and
cache fixed-size representations for a list of items.  
`ItemCacher` itself is extremely thin – all heavy lifting happens inside
the pager – but it plugs the pager into the generic
`BaseCacher` interface which is used throughout the project.

Workflow
--------
1) The client calls `ItemCacher.cache(contents)`.
2) `_cache()` is invoked (template method pattern inherited from
   `BaseCacher`).
3) `_cache()` creates an appropriately initialised `FastItemPager`,
   runs it, and finally returns the tensor that the pager produced.
4) `BaseCacher` stores that tensor in `self.repr` and marks the cache as
   valid (`self.cached = True`).

The concrete “operator” must implement the following attributes /
methods (checked by `FastItemPager`):
    • operator.inputer                     – instance of `BaseInputer`
    • operator.get_full_placeholder(n)     – allocates the destination
                                             tensor with shape (n, hidden)
    • operator(...)                        – callable that maps the
                                             stacked features to a hidden
                                             representation.
"""

from __future__ import annotations

from loader.cacher.base_cacher import BaseCacher
from loader.pager.fast_item_pager import FastItemPager


class ItemCacher(BaseCacher):
    """
    Thin wrapper around `FastItemPager`.

    All constructor arguments are forwarded to `BaseCacher`; see its
    doc-string for the full list of supported keyword parameters.
    """

    # Nothing to add here – we only rely on BaseCacher’s __init__
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------ #
    # BaseCacher override                                                #
    # ------------------------------------------------------------------ #
    def _cache(self, contents):
        """
        Build the item cache for the given *contents*.

        The method
          1) requests a zero-initialised placeholder from `self.operator`
             with the correct first dimension,
          2) instantiates a `FastItemPager` that will fill this buffer,
          3) executes the pager, and
          4) returns the populated tensor (→ stored in `self.repr`).

        Parameters
        ----------
        contents : list[Any]
            Sequence of items for which representations should be
            computed.

        Returns
        -------
        torch.Tensor
            Cached item representations with shape
            (len(contents), hidden_size).
        """
        # Size of the batch dimension equals number of items to cache
        item_size = len(contents)

        # Ask the operator to create the destination tensor
        placeholder = self.operator.get_full_placeholder(item_size)

        # ------------------------------------------------------------------
        # Set up the pager that actually computes the representations
        # ------------------------------------------------------------------
        pager = FastItemPager(
            inputer=self.operator.inputer,   # how to obtain embeddings
            contents=contents,              # the items themselves
            model=self.operator,            # callable that produces hidden states
            page_size=self.page_size,       # batch size
            hidden_size=self.hidden_size,   # dimensionality of output
            placeholder=placeholder,        # pre-allocated destination
        )

        # Trigger the batch-wise processing loop
        pager.run()

        # `FastItemPager` writes directly into `placeholder`
        # and also exposes it via `.fast_item_repr`.
        return pager.fast_item_repr
