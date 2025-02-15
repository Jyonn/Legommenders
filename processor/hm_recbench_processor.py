from processor.recbench_processor import RecBenchProcessor


class HMRBProcessor(RecBenchProcessor):
    IID_COL = 'article_id'
    UID_COL = 'customer_id'
    HIS_COL = 'history'
    LBL_COL = 'click'

    REQUIRE_STRINGIFY = True
    PROMPT = 'Here is a fashionable garment. '

    @property
    def attrs(self) -> dict:
        return dict(detail_desc=0)
