from processor.recbench_processor import RecBenchProcessor


class GoodreadsRBProcessor(RecBenchProcessor):
    IID_COL = 'bid'
    UID_COL = 'uid'
    HIS_COL = 'history'
    LBL_COL = 'click'

    REQUIRE_STRINGIFY = True
    PROMPT = 'Here is a book. '

    @property
    def attrs(self) -> dict:
        return dict(title=0)
