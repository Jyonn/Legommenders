"""
dataset.py

Thin wrapper that adapts a `LegoUT` object (our UniTok-based data
container) to the standard PyTorch `torch.utils.data.Dataset` API.

Key responsibilities
--------------------
1) **Indexing** – delegates `__getitem__` calls to the underlying
   `LegoUT` instance but returns *shallow copies* of each field so that
   later in the pipeline every sample can be modified in-place without
   affecting the cached version inside `LegoUT`.

2) **Optional resampling / augmentation** – an arbitrary callable
   (`resampler`) can be provided that takes a *sample-dict* and returns a
   (possibly transformed) one.  Typical use-cases are negative sampling,
   data augmentation or on-the-fly label smoothing.

3) **Iteration & length** – convenience wrappers around the base class
   so that the object works with every PyTorch DataLoader out of the box.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, Iterator, Any

from torch.utils.data import Dataset as BaseDataset

from loader.ut.lego_ut import LegoUT   # domain-specific UniTok wrapper


class DataSet(BaseDataset):
    """
    PyTorch-compatible dataset that wraps a `LegoUT` instance.

    Parameters
    ----------
    ut : LegoUT
        Tokenised dataset produced by UniTok (or a project-specific
        subclass thereof).
    resampler : Callable[[dict], dict] | None
        Optional transformation applied to every sample after copying it
        from `ut`.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        ut: LegoUT,
        resampler: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
    ) -> None:
        self.ut: LegoUT = ut
        self.resampler = resampler

    # ------------------------------------------------------------------ #
    # Dataset protocol                                                   #
    # ------------------------------------------------------------------ #
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Fetch a single example by index.

        Notes
        -----
        • We build a *shallow* copy (`copy.copy`) for every column so that
          list / tensor objects inside the sample remain independent.  
          This guards against subtle bugs where downstream components
          mutate the returned object (e.g. padding, masking) which would
          otherwise modify the cached version in `LegoUT`.
        """
        _sample = self.ut[index]            # raw sample from LegoUT
        sample: Dict[str, Any] = {}

        # Copy each column individually (keeps tensors as views but
        # duplicates lists / dicts etc.)
        for col in _sample:
            sample[col] = copy.copy(_sample[col])

        # Optional augmentation / negative sampling
        if self.resampler:
            sample = self.resampler(sample)

        return sample

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.ut)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Yield samples one by one.  Implemented explicitly so that we can
        rely on the copy & resampler logic of `__getitem__`.
        """
        for i in range(len(self)):
            yield self[i]
