class ColumnMap:
    def __init__(
            self,
            clicks_col: str = 'history',
            candidate_col: str = 'nid',
            label_col: str = 'label',
    ):
        self.clicks_col = clicks_col
        self.candidate_col = candidate_col
        self.label_col = label_col
        self.clicks_mask_col = '__clicks_mask__'
