from processor.recbench_processor import RecBenchProcessor


class YelpRBProcessor(RecBenchProcessor):
    UID_COL = 'user_id'
    IID_COL = 'business_id'
    HIS_COL = 'history'
    LBL_COL = 'click'

    REQUIRE_STRINGIFY = False
    PROMPT = 'Here is a restaurant. '

    @property
    def attrs(self) -> dict:
        return dict(name=0, address=0, city=0, state=0)
