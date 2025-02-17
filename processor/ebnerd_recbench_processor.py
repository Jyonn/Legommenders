from processor.recbench_processor import RecBenchProcessor


class EBNeRDRBProcessor(RecBenchProcessor):
    IID_COL = 'nid'
    UID_COL = 'uid'
    HIS_COL = 'history'
    LBL_COL = 'click'

    REQUIRE_STRINGIFY = False
    PROMPT = 'Here is a piece of news. '

    @property
    def attrs(self) -> dict:
        return dict(title=0, subtitle=0, body=0, category=0)
