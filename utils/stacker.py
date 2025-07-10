"""
stacker.py

A small set of helper classes that aggregate / “stack” batches of
heterogeneously structured data—tensors, numbers, nested dictionaries,
etc.—into batched tensors or lists.

Why?
----
In many data-loading pipelines we read one *example* at a time.  When we
assemble a batch we want to merge N samples that might themselves be
nested dictionaries.  The standard PyTorch `default_collate` handles
tensors and lists but quickly becomes cumbersome for deeply nested,
custom structures.

`Stacker` addresses this by:

1. **Recursively building a prototype** of the sample structure.
2. Collecting an array of values for every leaf node.
3. Calling an *aggregator* function (default: `torch.stack`) on each leaf
   to create the final batched representation.

There are three concrete classes:

Stacker
    Full recursive stacking for arbitrarily nested dicts.

OneDepthStacker
    Only supports a single dict level (no recursion), but simpler & faster.

FastStacker
    Identical to `Stacker` but caches the prototype so that repeated calls
    on the same data structure avoid the prototype construction cost.

Usage
-----
>>> batch = [ sample1, sample2, ..., sampleN ]      # list of dicts
>>> stacker = Stacker(torch.stack)
>>> collated = stacker(batch)
"""

import copy
from collections import OrderedDict
from typing import Callable, Union, List, Dict, Any

import torch


# =============================================================================
#                                  Stacker
# =============================================================================
class Stacker:
    """
    Parameters
    ----------
    aggregator : Callable, default=torch.stack
        A function that takes a *list* of leaf values (e.g., tensors) and
        returns an aggregated/stacked object (e.g., a batched tensor).
        Can be replaced with `np.stack`, `lambda x: x` (no-op), etc.
    """

    def __init__(self, aggregator: Callable = None):
        self.aggregator = aggregator or torch.stack

    # ---------------------------------------------------------------------
    # Helper: create an *empty* container with the same structure
    # ---------------------------------------------------------------------
    def _build_prototype(self, item: Any) -> Union[dict, list]:
        """
        Recursively walk through `item` and construct an *empty* container
        (dict / list) that mirrors its structure but with lists in place of
        every leaf.  The lists will later collect values from N samples.
        """
        # Base-case: leaf node → return an empty list that will hold values
        if not isinstance(item, dict):
            return []

        # Recursive case: dict → dive deeper
        prototype: Dict[str, Union[dict, list]] = {}
        for k in item.keys():
            if isinstance(item[k], dict):
                prototype[k] = self._build_prototype(item[k])
            else:
                prototype[k] = []
        return prototype

    # ---------------------------------------------------------------------
    # Helper: append values from *one* sample into the prototype collector
    # ---------------------------------------------------------------------
    def _insert_data(self, prototype: Union[dict, list], item: Any):
        """
        Fill the prototype container with data from a *single* sample by
        appending the corresponding value into every leaf list.
        """
        if not isinstance(item, dict):
            prototype.append(item)
            return

        for k in item.keys():
            if isinstance(item[k], dict):
                self._insert_data(prototype[k], item[k])
            else:
                prototype[k].append(item[k])

    # ---------------------------------------------------------------------
    # Helper: apply the aggregator on every leaf list
    # ---------------------------------------------------------------------
    def _aggregate(self, prototype: Union[dict, list], apply: Callable = None):
        """
        Replace lists of collected values with the result of
        `self.aggregator`.  Optionally, run an extra `apply` function
        afterwards (e.g., `.float()` on tensors).
        """
        if isinstance(prototype, list):
            prototype = self.aggregator(prototype)
            if apply:
                prototype = apply(prototype)
            return prototype

        # Recurse for dictionaries
        for k in prototype.keys():
            if isinstance(prototype[k], dict):
                self._aggregate(prototype[k], apply=apply)
            else:
                prototype[k] = self.aggregator(prototype[k])
                if apply:
                    prototype[k] = apply(prototype[k])
        return prototype

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def stack(self, item_list: List[Any], apply: Callable = None):
        """
        The main entry point.  Takes a list of N samples, builds a structure
        prototype, inserts data, calls the aggregator, and returns the
        collated batch.

        Parameters
        ----------
        item_list : list
            A list of *samples* (dicts / objects) that share the same layout.
        apply : Callable, optional
            Extra function to run **after** aggregation on every leaf
            (common use-case: type casting).
        """
        prototype = self._build_prototype(item_list[0])
        for item in item_list:
            self._insert_data(prototype, item)
        if self.aggregator:
            prototype = self._aggregate(prototype, apply=apply)
        return prototype

    # Make the object directly callable for convenience
    def __call__(self, item_list: list, apply: Callable = None):
        return self.stack(item_list, apply=apply)


# =============================================================================
#                            OneDepthStacker
# =============================================================================
class OneDepthStacker(Stacker):
    """
    Same interface as `Stacker` but assumes that each sample is a *flat*
    dictionary (no nested dicts).  Prototype creation becomes simpler.
    """

    def __init__(self, aggregator: Callable = None):
        super().__init__(aggregator)

    def _build_prototype(self, item):
        if not isinstance(item, dict):
            return []

        prototype = OrderedDict()
        for k in item.keys():
            prototype[k] = []
        return prototype


# =============================================================================
#                              FastStacker
# =============================================================================
class FastStacker(Stacker):
    """
    Drop-in replacement for `Stacker` that caches the prototype after the
    first call, making subsequent `stack` operations faster if the sample
    structure does not change between batches.
    """

    def __init__(self, aggregator: Callable = None):
        super().__init__(aggregator)
        self.prototype = None  # type: Union[dict, list, None]

    def stack(self, item_list: list, apply: Callable = None):
        # Build & cache prototype only once
        if self.prototype is None:
            self.prototype = self._build_prototype(item_list[0])

        prototype = copy.deepcopy(self.prototype)  # fresh copy for every batch
        for item in item_list:
            self._insert_data(prototype, item)

        if self.aggregator:
            prototype = self._aggregate(prototype, apply=apply)
        return prototype


# =============================================================================
#                               Quick demo
# =============================================================================
if __name__ == "__main__":
    stacker = Stacker(torch.stack)

    sample_a = torch.tensor([1, 2, 3])
    print("Stacked:", stacker.stack([sample_a]))
