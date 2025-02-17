from processor.recbench_processor import RecBenchProcessor


class NetflixRBProcessor(RecBenchProcessor):
    IID_COL = 'mid'
    UID_COL = 'uid'
    HIS_COL = 'history'
    LBL_COL = 'click'

    REQUIRE_STRINGIFY = True
    PROMPT = 'Here is a movie. '

    @property
    def attrs(self) -> dict:
        return dict(title=0)
