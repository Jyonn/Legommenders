"""
slice_dict.py

Utility classes that make ordinary dictionaries *sliceable* in a NumPy-/Pandas-like
fashion.

Motivation
----------
Very often we store *aligned* lists / tensors under different keys:

    batch = {
        "user_id": [  4,   8, 15, 16, 23],
        "item_id": [ 42, 108,  81,  11,  19],
        "label"  : [  1,   0,   1,   0,   1],
    }

If we want to take only the first three examples we usually have to loop over
every key manually:

    sliced = {k: v[:3] for k, v in batch.items()}

`SliceDict` (and its ordered counterpart `SliceOrderedDict`) implement `__getitem__`
so that a simple `batch[:3]` performs the same operation and returns *the same
sub-class* with all keys sliced consistently.

Classes
-------
SliceDict
    Drop-in replacement for the built-in `dict` with slice support.

SliceOrderedDict
    Same as above but inherits from `collections.OrderedDict` to preserve key order.
"""

from collections import OrderedDict
from typing import Any, Dict, TypeVar, Union

T = TypeVar("T")  # Generic type for values stored in the dict


# =============================================================================
#                                 SliceDict
# =============================================================================
class SliceDict(dict):
    """
    A dict that supports slicing across *all* values simultaneously.

    Examples
    --------
    >>> d = SliceDict(a=[1, 2, 3], b=[10, 20, 30])
    >>> d[1:]                       # SliceDict({'a': [2, 3], 'b': [20, 30]})
    >>> d["a"]                      # Standard key access still works
    """

    def __getitem__(self, item: Union[str, slice]) -> Union[Any, "SliceDict"]:
        # If the item is NOT a slice, fall back to the normal dict behaviour
        if not isinstance(item, slice):
            return super(SliceDict, self).__getitem__(item)

        # Otherwise, build a *new* SliceDict containing sliced values
        slice_dict = SliceDict()
        for k, v in self.items():
            slice_dict[k] = v[item]      # Delegate the actual slicing to list/np.array/torch.Tensor
        return slice_dict


# =============================================================================
#                              SliceOrderedDict
# =============================================================================
class SliceOrderedDict(OrderedDict):
    """
    Same idea as `SliceDict` but keeps keys in insertion order.

    Useful when deterministic ordering is important (e.g. serialisation).
    """

    def __getitem__(self, item: Union[str, slice]) -> Union[Any, "SliceOrderedDict"]:
        if not isinstance(item, slice):
            return super(SliceOrderedDict, self).__getitem__(item)

        slice_dict = SliceOrderedDict()
        for k, v in self.items():
            slice_dict[k] = v[item]
        return slice_dict


# =============================================================================
#                                  Demo
# =============================================================================
if __name__ == "__main__":
    # Quick self-test / demonstration
    d = SliceDict(
        a=[1, 2, 3],
        b=[4, 5, 6],
        c=[7, 8, 9],
    )
    print("Original:", d)
    print("d[1:]:   ", d[1:])          # Slice the last two elements for every key

    # Verify that standard dict operations remain unchanged
    original = dict(a=1, b=2)
    copy_dict = dict(original)
    print("Regular dict copy still works:", copy_dict)
