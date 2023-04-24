class ColumnMap:
    def __init__(
            self,
            clicks_col: str = 'history',
            candidate_col: str = 'nid',
            label_col: str = 'click',
            neg_col: str = 'neg',
            group_col: str = 'imp',
            user_col: str = 'uid',
            index_col: str = 'index',
            fake_col: str = None,
            **kwargs
    ):
        self.clicks_col = clicks_col
        self.candidate_col = candidate_col
        self.label_col = label_col
        self.neg_col = neg_col
        self.group_col = group_col
        self.user_col = user_col
        self.index_col = index_col
        self.fake_col = fake_col
        self.clicks_mask_col = '__clicks_mask__'
