from typing import Optional, Callable


class BaseCacher:
    def __init__(self, operator, page_size, hidden_size, activate=True, trigger=None):
        self.operator = operator
        self.page_size = page_size
        self.hidden_size = hidden_size
        self.trigger: Optional[Callable] = trigger

        self.cached: bool
        self._set_cached(False)

        self._activate = activate
        self.repr = None

        self.current = dict()
        self.current_count = self.cache_count = 0

    def _set_cached(self, cached):
        self.cached = cached
        self.trigger(cached)

    def _cache(self, contents):
        raise NotImplementedError

    def cache(self, contents):
        self.clean()

        if not self._activate:
            return

        self.repr = self._cache(contents)
        self._set_cached(True)

    def clean(self):
        self.repr = None
        self._set_cached(False)
