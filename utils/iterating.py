class Iterating:
    def __init__(self, list_types=None):
        self.list_types = list_types or [list, tuple, set]

    def worker_dict(self, d: dict):
        return {k: self.worker(d[k]) for k in d}

    def worker_list(self, l: list):
        return [self.worker(x) for x in l]

    def is_list(self, x):
        for t in self.list_types:
            if isinstance(x, t):
                return True
        return False

    def custom_worker(self, x):
        raise NotImplementedError

    def worker(self, x):
        if isinstance(x, dict):
            return self.worker_dict(x)
        elif self.is_list(x):
            return self.worker_list(x)
        return self.custom_worker(x)
