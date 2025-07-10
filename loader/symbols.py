"""
symbols.py

A very small helper module that defines *singleton* symbols
acting as “named constants” throughout the project.

Instead of storing raw strings everywhere (which are prone to typos and
hard-to-track bugs) we rely on the `unitok.Symbol` class that guarantees
uniqueness and identity comparison:

    from symbols import Symbols

    if phase is Symbols.train:            # identity comparison is safe
        ...

Why not just use plain strings?
--------------------------------
With strings one can accidentally create two different objects that have
the same text but are *not* identical by reference.  `unitok.Symbol`
creates a single immutable Python object for each distinct name, which
means `a is b` is `True` if and only if both variables refer to the same
logical symbol, providing an additional safety guard over the usual
string equality check.
"""

from unitok import Symbol


class Symbols:
    """
    A collection of *static* attributes, each representing a distinct phase
    or flag within the code base.

    Access pattern:
        Symbols.train      # → <Symbol 'train'>
        Symbols.best       # → <Symbol 'best'>
    """

    # ---------------------------------------------------------------------
    # Dataset / running phases
    # ---------------------------------------------------------------------
    train = Symbol("train")          # training split / phase
    dev = Symbol("dev")              # validation split / phase
    test = Symbol("test")            # test split / phase
    fast_eval = Symbol("fast_eval")  # a lightweight evaluation phase (e.g., for debugging)

    # ---------------------------------------------------------------------
    # Early-stopping & control flags
    # ---------------------------------------------------------------------
    best = Symbol("best")            # indicates the best model checkpoint so far
    skip = Symbol("skip")            # signal to skip current iteration / batch
    stop = Symbol("stop")            # signal to terminate training loop

    # ---------------------------------------------------------------------
    # Entity types
    # ---------------------------------------------------------------------
    user = Symbol("user")            # user entity identifier
    item = Symbol("item")            # item entity identifier
