"""
ut_hub.py

Very small *singleton-like* cache for `LegoUT` objects.

Loading a UniTok dataset from disk can be costly because it involves
reading multiple pickle / numpy files and rebuilding internal indices.
`UTHub` therefore keeps one in-memory instance **per path** so that
subsequent calls asking for the *same* dataset receive the already
initialized object instead of triggering another load.

Usage
-----
>>> ut_train = UTHub.get("./data/train_ut", use_filter_cache=True)
>>> ut_again  = UTHub.get("./data/train_ut")      # returns *exact* same instance
assert ut_train is ut_again
"""

from loader.ut.lego_ut import LegoUT


class UTHub:
    """
    Simple registry that maps `path -> LegoUT` and only loads each path
    once during the lifetime of the Python process.
    """

    # class-level storage for the cached datasets
    _instances: dict[str, LegoUT] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    @classmethod
    def get(cls, path: str, use_filter_cache: bool = False) -> LegoUT:
        """
        Retrieve a `LegoUT` instance for *path*.  If the dataset was
        requested earlier the cached object is returned; otherwise it is
        loaded from disk.

        Parameters
        ----------
        path : str
            Directory that contains a dataset previously saved by UniTok.
        use_filter_cache : bool
            Forwarded to `LegoUT.load`.  Controls whether filter results
            should be cached / restored (see `LegoUT` docs for details).
        """
        # Return cached instance if available
        if path in cls._instances:
            return cls._instances[path]

        # Otherwise load from disk and remember it
        depot = LegoUT.load(path, use_filter_cache=use_filter_cache)
        cls._instances[path] = depot
        return depot
