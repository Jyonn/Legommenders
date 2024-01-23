class BaseCacher:
    def __init__(self, operator, page_size, hidden_size, activate=True):
        self.operator = operator
        self.page_size = page_size
        self.hidden_size = hidden_size

        self._activate = activate
        self.cached = False
        self.repr = None

        self.current = dict()
        self.current_count = self.cache_count = 0

    def _cache(self, contents):
        raise NotImplementedError

    def cache(self, contents):
        self.clean()

        if not self._activate:
            return

        self.repr = self._cache(contents)
        self.cached = True

    def clean(self):
        self.cached = False
        self.repr = None
