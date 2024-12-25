from typing import Union

from unitok import UniTok
from oba import Obj

from loader.depot.ut_hub import UTHub


class DataHub:
    def __init__(
            self,
            ut: Union[UniTok, str],
            order,
            append=None,
    ):
        self.ut = ut if isinstance(ut, UniTok) else UTHub.get(ut)
        self.order = Obj.raw(order)
        self.append = Obj.raw(append) or []

        # self.depot.select_cols(self.order + self.append)
