"""
base_cacher.py

Very small utility base-class that standardises the *caching* workflow
used by several pagers / helpers inside the project.

Responsibilities
----------------
1) Hold the final cache in `self.repr`.
2) Keep book on how many elements have been accumulated so far
   (`self.current_count`, `self.cache_count`).
3) Fire an *optional* `trigger` callback whenever the cache becomes
   valid / invalid (`self.cached` flag).

The concrete caching logic itself is delegated to the sub-class via
`_cache()`.
"""

from __future__ import annotations
from typing import Optional, Callable, Any, Sequence


class BaseCacher:
    """
    Base class that manages the life-cycle of a representation cache.

    Parameters
    ----------
    operator : Any
        Arbitrary object needed by the concrete cacher implementation
        (e.g. a neural network or feature extractor).
    page_size : int
        Number of items that should be processed in one mini-batch.
        Only stored here — the sub-class decides how to make use of it.
    hidden_size : int
        Dimensionality of one cached representation element.
    activate : bool, default=True
        If *False* the cacher becomes a no-op; this is useful when
        toggling expensive pre-computation during debugging.
    trigger : Callable[[bool], None], optional
        Callback that is executed whenever the internal `cached` flag
        changes.  Receives the new flag value as its only argument.
    """

    # ------------------------------------------------------------------ #
    # Constructor                                                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        operator: Any,
        page_size: int,
        hidden_size: int,
        activate: bool = True,
        trigger: Optional[Callable[[bool], None]] = None,
    ):
        self.operator = operator
        self.page_size = page_size
        self.hidden_size = hidden_size
        self.trigger = trigger or (lambda *_: None)  # fall-back no-op

        # Flag that indicates whether `self.repr` currently holds valid
        # data.  Use the helper `_set_cached` to update it!
        self.cached: bool
        self._set_cached(False)

        # Switch that can completely disable the cacher
        self._activate = activate

        # Container for the final cached representation
        self.repr: Optional[Any] = None

        # Temporary holders used by sub-classes while accumulating data
        self.current: dict[str, list[Any]] = {}
        self.current_count: int = 0
        self.cache_count: int = 0

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _set_cached(self, cached: bool) -> None:
        """
        Update the `cached` flag and fire the `trigger` callback.
        """
        self.cached = cached
        self.trigger(cached)

    # ------------------------------------------------------------------ #
    # Methods intended to be overridden / called by sub-classes          #
    # ------------------------------------------------------------------ #
    def _cache(self, contents: Sequence[Any]):
        """
        Concrete caching routine that must be implemented by each
        subclass.  It should return the final representation that will
        be stored in `self.repr`.

        Parameters
        ----------
        contents : Sequence
            Collection of items that need to be cached.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def cache(self, contents: Sequence[Any]) -> None:
        """
        Main entry point: (re)build the cache for the supplied contents.

        The method
          • clears any previous cache (`clean`)
          • optionally aborts when the cacher is de-activated
          • delegates the heavy lifting to `_cache`
          • marks the cache as *valid* afterwards
        """
        self.clean()

        # Early exit when caching is disabled
        if not self._activate:
            return

        self.repr = self._cache(contents)
        self._set_cached(True)

    def clean(self) -> None:
        """
        Remove the current cache and mark it as *invalid*.
        """
        self.repr = None
        self._set_cached(False)
