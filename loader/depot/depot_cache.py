from UniTok import UniDep

from loader.depot.fc_unidep import FCUniDep


class DepotCache:
    depots = dict()

    @classmethod
    def get(cls, path, filter_cache=False) -> UniDep:
        if path in cls.depots:
            return cls.depots[path]
        cls.depots[path] = FCUniDep(path, filter_cache=filter_cache)
        return cls.depots[path]
