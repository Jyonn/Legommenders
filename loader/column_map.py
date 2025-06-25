from unitok import UniTok


class ColumnMap:
    def __init__(
            self,
            history_col: str = 'history',
            item_col: str = 'item_id',
            label_col: str = 'click',
            user_col: str = 'user_id',
            group_col: str = 'user_id',
            neg_col: str = None,
    ):
        self.history_col = history_col
        self.item_col = item_col
        self.label_col = label_col
        self.group_col = group_col
        self.user_col = user_col
        self.neg_col = neg_col
        self.mask_col = '__clicks_mask__'

        self.history_vocab = None
        self.item_vocab = None
        self.label_vocab = None
        self.user_vocab = None
        self.group_vocab = None

    def set_column_vocab(self, inter_ut: UniTok):
        def col_to_vocab(col: str):
            return inter_ut.meta.features[col].tokenizer.vocab.name

        self.history_vocab = col_to_vocab(self.history_col)
        self.item_vocab = col_to_vocab(self.item_col)
        self.label_vocab = col_to_vocab(self.label_col)
        self.user_vocab = col_to_vocab(self.user_col)
        self.group_vocab = col_to_vocab(self.group_col)
