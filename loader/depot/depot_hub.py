from UniTok import UniDep

from loader.depot.caching_depot import CachingDep


class DepotHub:
    depots = dict()

    @classmethod
    def get(cls, path, filter_cache=False) -> UniDep:
        if path in cls.depots:
            return cls.depots[path]
        cls.depots[path] = CachingDep(path, filter_cache=filter_cache)
        return cls.depots[path]
