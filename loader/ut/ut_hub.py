from loader.ut.lego_ut import LegoUT


class UTHub:
    _instances = dict()

    @classmethod
    def get(cls, path, use_filter_cache=False) -> LegoUT:
        if path in cls._instances:
            return cls._instances[path]

        depot = LegoUT.load(path, use_filter_cache=use_filter_cache)
        cls._instances[path] = depot

        return depot
