from typing import Union

from UniTok import UniDep
from oba import Obj

from loader.depot.depot_cache import DepotCache


class NRDepot:
    def __init__(
            self,
            depot: Union[UniDep, str],
            order,
            append=None,
    ):
        self.depot = depot if isinstance(depot, UniDep) else DepotCache.get(depot)
        self.order = Obj.raw(order)
        self.append = Obj.raw(append) or []
