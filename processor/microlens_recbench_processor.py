from processor.recbench_processor import RecBenchProcessor


class MicroLensRBProcessor(RecBenchProcessor):
    IID_COL = 'item'
    UID_COL = 'user'
    HIS_COL = 'history'
    LBL_COL = 'click'

    REQUIRE_STRINGIFY = True
    PROMPT = 'Here is a micro video. '

    @property
    def attrs(self) -> dict:
        return dict(title=0)
