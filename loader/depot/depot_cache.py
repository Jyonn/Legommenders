from UniTok import UniDep


class DepotCache:
    depots = dict()

    @classmethod
    def get(cls, path) -> UniDep:
        if path in cls.depots:
            return cls.depots[path]
        cls.depots[path] = UniDep(path)
        return cls.depots[path]
