from UniTok import UniDep

from loader.depot.caching_depot import CachingDep


class DepotHub:
    depots = dict()

    @classmethod
    def get(cls, path, filter_cache=False) -> UniDep:
        if path in cls.depots:
            return cls.depots[path]

        depot = CachingDep(path, filter_cache=filter_cache)
        # depot.deep_union(True)
        cls.depots[path] = depot

        return depot
