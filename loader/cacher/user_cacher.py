"""
user_cacher.py

Thin wrapper that employs `FastUserPager` to pre-compute and cache
representations for all users.  Mirrors the implementation of
`ItemCacher` but works with *user* features.

Design
------
The actual computation is delegated to `FastUserPager`.  
`UserCacher` only fulfils the `BaseCacher` protocol:

    • `_cache()` has to receive a collection of `contents`,
      launch the pager, and return the final tensor.
    • The returned tensor is then stored in `self.repr`
      by the parent class (`BaseCacher`).

Specialty
---------
The destination tensor (`placeholder`) is **allocated outside** of the
pager and passed in via the constructor.  This is handy because the
total number of users is usually known upfront, so the memory can be
allocated once during initialisation instead of every caching run.
"""

from __future__ import annotations

from loader.cacher.base_cacher import BaseCacher
from loader.pager.fast_user_pager import FastUserPager


class UserCacher(BaseCacher):
    """
    Wrapper around `FastUserPager`.

    Parameters
    ----------
    placeholder : torch.Tensor
        Pre-allocated tensor with shape
            (num_users, hidden_size)
        that will receive the user representations.
    **kwargs
        Forwarded to `BaseCacher`.  Must at least include
            operator    – callable that maps a batch of features to a
                           hidden representation
            page_size   – int, mini-batch size of the pager
            hidden_size – int, dimensionality of each representation
    """

    # ------------------------------------------------------------------ #
    # Constructor                                                        #
    # ------------------------------------------------------------------ #
    def __init__(self, placeholder, **kwargs):
        # Let BaseCacher set up the generic bookkeeping
        super().__init__(**kwargs)

        # Keep a reference to the pre-allocated destination buffer
        self.placeholder = placeholder

    # ------------------------------------------------------------------ #
    # BaseCacher override                                                #
    # ------------------------------------------------------------------ #
    def _cache(self, contents):
        """
        Build the user cache for the supplied *contents*.

        1) Instantiate a `FastUserPager` that will fill `self.placeholder`
           with the model outputs.
        2) Execute the pager.
        3) Return the populated tensor (stored in `self.repr` by caller).

        Parameters
        ----------
        contents : list[Any]
            Collection of user feature dictionaries.

        Returns
        -------
        torch.Tensor
            Cached user representations with shape identical to
            `self.placeholder`.
        """
        # Create the pager responsible for batch-wise processing
        pager = FastUserPager(
            contents=contents,          # list of user feature dicts
            model=self.operator,        # callable → hidden states
            page_size=self.page_size,   # batch size for paging
            hidden_size=self.hidden_size,
            placeholder=self.placeholder,  # destination buffer
        )

        # Run the paging loop
        pager.run()

        # The pager wrote directly into `placeholder` and exposes it
        # again via `.fast_user_repr`.
        return pager.fast_user_repr
