from processor.recbench_processor import RecBenchProcessor


class PogRBProcessor(RecBenchProcessor):
    IID_COL = 'item_id'
    UID_COL = 'user_id'
    HIS_COL = 'history'
    LBL_COL = 'click'

    REQUIRE_STRINGIFY = False
    PROMPT = 'Here is a fashion outfit. '

    @property
    def attrs(self) -> dict:
        return dict(title_en=0)
