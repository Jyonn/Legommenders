from loader.depot.caching_ut import CachingUT


class UTHub:
    depots = dict()

    @classmethod
    def get(cls, path, filter_cache=False) -> CachingUT:
        if path in cls.depots:
            return cls.depots[path]

        depot = CachingUT.load(path, filter_cache=filter_cache)
        # depot.deep_union(True)
        cls.depots[path] = depot

        return depot
