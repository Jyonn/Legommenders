from processor.recbench_processor import RecBenchProcessor


class AutomotiveRBProcessor(RecBenchProcessor):
    UID_COL = 'reviewerID'
    IID_COL = 'asin'
    HIS_COL = 'history'
    LBL_COL = 'click'

    REQUIRE_STRINGIFY = False
    PROMPT = 'Here is a car. '

    @property
    def attrs(self) -> dict:
        return dict(title=0)
