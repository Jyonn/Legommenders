"""
lego_ut.py

Extension of `unitok.UniTok` that adds **filter-result caching** and a
few convenience helpers tailored for the *Legommenders* project.

Motivation
----------
Applying complex filter pipelines to large tokenised datasets can be
time-consuming.  
`LegoUT` therefore persists the list of filter functions **and** the
resulting index set (`_legal_indices`) to disk so that the same filter
combination can be restored instantly the next time the dataset is
loaded—if `use_filter_cache=True`.

Core additions compared to vanilla UniTok
-----------------------------------------
1) Persistent filter cache located at  <save_dir>/filters/
   • meta-file  : *filter_cache.pkl*  
                 – list[dict] describing the filter configuration
   • data files : <random>.pkl  – pickled list of surviving indices

2) Attribute selection / truncation utilities (`rebuild`) that allow
   downstream tasks to request only a subset of the available columns.

3) Override of `__getitem__` so that the selected-attribute view is taken
   into account automatically.

Naming convention for the cache dictionaries
--------------------------------------------
`gft` : Symbol("general_filters")          – list[str] (no column given)
`fft` : Symbol("feature_specific_filters") – dict[col, list[str]]
`pth` : Symbol("path")                     – file name of the pickled
                                            `_legal_indices`
"""

from __future__ import annotations

import os.path
from typing import Any, Dict, List, Tuple

from unitok import Symbol, UniTok
from pigmento import pnt

from utils import function, io


class LegoUT(UniTok):
    """
    Project-specific subclass of `UniTok` with filter-result caching.

    Public usage pattern
    --------------------
    >>> ut = LegoUT.load("/path/to/dataset", use_filter_cache=True)
    >>> ut.filter("lambda x: x['age'] > 20")
    >>> ut.filter("lambda x: x > 0", col="label")
    """

    # ---------------- #
    # Class attributes #
    # ---------------- #
    # (They are declared here only for the benefit of IDE type checkers;
    #  the actual values are populated in `load`.)
    _use_filter_cache: bool
    _general_filters: List[str]
    _feature_specific_filters: Dict[str, List[str]]
    _filter_cache_dir: str
    _filter_cache_meta_path: str
    _caches: List[Dict[str, Any]]
    _selected_attrs: Tuple[str, ...]

    # keys used inside the cache dictionaries
    gft = Symbol("general_filters")
    fft = Symbol("feature_specific_filters")
    pth = Symbol("path")

    # ------------------------------------------------------------------ #
    # Construction / loading                                             #
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, save_dir: str, use_filter_cache: bool = False) -> "LegoUT":
        """
        Load a previously saved UniTok dataset and wrap it in a LegoUT
        that supports filter caching.

        Parameters
        ----------
        save_dir : str
            Directory returned by `UniTok.save` earlier.
        use_filter_cache : bool
            If True, LegoUT will look for cached filter results and reuse
            them when possible.
        """
        # Let UniTok handle the heavy lifting
        ut: LegoUT = super().load(save_dir)        # type: ignore[assignment]
        ut._use_filter_cache = use_filter_cache

        # Runtime filter status (will be filled when `filter` is called)
        ut._general_filters = []
        ut._feature_specific_filters = {}

        # Where the cache files live
        ut._filter_cache_dir = os.path.join(save_dir, "filters")
        os.makedirs(ut._filter_cache_dir, exist_ok=True)

        ut._filter_cache_meta_path = os.path.join(
            ut._filter_cache_dir, "filter_cache.pkl"
        )

        ut._caches = []            # list of previously stored caches
        ut._load_cache()           # populate the list if files exist

        ut._selected_attrs = set()

        return ut

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _filter_equals(self, other: Dict[str, Any]) -> bool:
        """
        Check whether `other` (one entry from the meta-file) describes
        *exactly* the same filter configuration as the one currently
        stored in the instance variables.
        """
        # Early out if mandatory keys are missing
        if (
            self.gft.name not in other
            or self.fft.name not in other
            or self.pth.name not in other
        ):
            return False

        general_filters = other[self.gft.name]
        feature_specific_filters = other[self.fft.name]

        # Compare general (column-agnostic) filters
        if len(general_filters) != len(self._general_filters):
            return False
        for func in general_filters:
            if func not in self._general_filters:
                return False

        # Compare column-specific filters
        if len(feature_specific_filters) != len(self._feature_specific_filters):
            return False
        for feature_name, func_list in feature_specific_filters.items():
            if feature_name not in self._feature_specific_filters:
                return False
            if len(func_list) != len(self._feature_specific_filters[feature_name]):
                return False
            for func in func_list:
                if func not in self._feature_specific_filters[feature_name]:
                    return False

        return True

    # ------------------------------------------------------------------ #
    # Cache management                                                   #
    # ------------------------------------------------------------------ #
    def _store_cache(self) -> None:
        """Persist the current `_legal_indices` together with its meta."""
        pnt(f"store filter cache on {str(self)}")

        # Give the data file a random name to avoid collisions
        filter_name = f"{function.get_random_string(6)}.pkl"
        filter_path = os.path.join(self._filter_cache_dir, filter_name)

        # Save index list
        io.pkl_save(self._legal_indices, filter_path)

        # Update meta information and save
        self._caches.append(
            {
                self.gft.name: self._general_filters,
                self.fft.name: self._feature_specific_filters,
                self.pth.name: filter_name,
            }
        )
        io.pkl_save(self._caches, self._filter_cache_meta_path)
        self._load_cache()     # reload so that file count is correct

    def _load_cache(self) -> None:
        """Populate `self._caches` if meta file exists."""
        self._caches = []
        if not self._use_filter_cache:
            return
        if os.path.exists(self._filter_cache_meta_path):
            self._caches = io.pkl_load(self._filter_cache_meta_path)
        pnt(f"load {len(self._caches)} filter caches on {str(self)}")

    # ------------------------------------------------------------------ #
    # Public API overrides                                               #
    # ------------------------------------------------------------------ #
    def filter(self, func: str, col: str | None = None) -> "LegoUT":
        """
        Apply a filter function (given as its *string representation*) to
        the dataset, optionally restricted to a single column.

        When caching is enabled, the method first tries to find a
        matching pre-computed index list.  If none is found, the base
        implementation (`UniTok.filter`) is executed and the result is
        cached for future runs.
        """
        if self._use_filter_cache:
            # ------------------------------------------------------------------
            # 1) Record the requested filter in our bookkeeping structures
            # ------------------------------------------------------------------
            if col is None:     # column-agnostic filter
                if func not in self._general_filters:
                    self._general_filters.append(func)
            else:               # column-specific
                if col not in self._feature_specific_filters:
                    self._feature_specific_filters[col] = []
                if func not in self._feature_specific_filters[col]:
                    self._feature_specific_filters[col].append(func)

            # ------------------------------------------------------------------
            # 2) Check whether an identical filter combination is cached
            # ------------------------------------------------------------------
            for cached_filter in self._caches:
                if self._filter_equals(cached_filter):
                    # bingo – load the stored indices and flags
                    path = os.path.join(
                        self._filter_cache_dir, cached_filter[self.pth.name]
                    )
                    self._legal_indices = io.pkl_load(path)
                    self._legal_flags = [False] * self._sample_size
                    for index in self._legal_indices:
                        self._legal_flags[index] = True
                    return self   # early return, no need to recompute

        # ------------------------------------------------------------------
        # 3) Cache miss or caching disabled – fall back to UniTok implementation
        # ------------------------------------------------------------------
        super().filter(eval(func), col)   # evaluate string to function

        # ------------------------------------------------------------------
        # 4) Store the newly computed result if caching is requested
        # ------------------------------------------------------------------
        if self._use_filter_cache:
            self._store_cache()

        return self

    # --------------------- #
    # Utility functionality #
    # --------------------- #
    def reset(self, data: Dict[str, Any]) -> None:
        """
        Replace `self.data` and reset internal indices.

        Useful when you want to plug pre-filtered or otherwise manipulated
        data frames back into the same LegoUT object.
        """
        self.data = data
        self.init_indices()    # UniTok helper

    def rebuild(self, selected_attrs: Dict[str, int | None]) -> None:
        """
        (Re)apply truncation on specified attributes and remember which
        attributes are currently selected.

        Parameters
        ----------
        selected_attrs : dict[str, int | None]
            Mapping `attribute_name -> truncate_length`.  
            If *truncate_length* evaluates to `False`/`0` no truncation is
            performed for that attribute.
        """
        attrs = set()
        for attr, truncate in selected_attrs.items():
            attrs.add(attr)
            if truncate:
                self.retruncate(attr, truncate)
        self._selected_attrs = tuple(attrs)

    # ------------------------------------------------------------------ #
    # Data access                                                        #
    # ------------------------------------------------------------------ #
    def __getitem__(self, item):
        """
        Forward to the base implementation but respect
        `self._selected_attrs` if it is non-empty.
        """
        if len(self._selected_attrs):
            return super().__getitem__((item, self._selected_attrs))
        return super().__getitem__(item)
