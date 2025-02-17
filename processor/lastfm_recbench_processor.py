from processor.recbench_processor import RecBenchProcessor


class LastfmRBProcessor(RecBenchProcessor):
    UID_COL = 'uid'
    IID_COL = 'tid'
    LBL_COL = 'click'
    HIS_COL = 'history'

    REQUIRE_STRINGIFY = True
    PROMPT = 'Here is a piece of music. '

    @property
    def attrs(self) -> dict:
        return dict(track_name=0, artist_name=0)
