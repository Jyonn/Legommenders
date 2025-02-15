from processor.recbench_processor import RecBenchProcessor


class CDsRBProcessor(RecBenchProcessor):
    UID_COL = 'reviewerID'
    IID_COL = 'asin'
    HIS_COL = 'history'
    LBL_COL = 'click'

    REQUIRE_STRINGIFY = False
    PROMPT = 'Here is a piece of music. '

    @property
    def attrs(self) -> dict:
        return dict(title=0)
