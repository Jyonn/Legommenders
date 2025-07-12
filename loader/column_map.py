"""
column_map.py

Utility class that serves as a **centralized schema description** for
datasets used throughout the project.  
It provides two complementary pieces of information:

1) Column names in the raw / tokenized data frame
   (`history_col`, `item_col`, …).

2) Names of the *vocabularies* that belong to those columns
   (to be filled in via `set_column_vocab` once a `UniTok` object is
   available).

Having a single object that knows both the *column* and the associated
*vocabulary* greatly simplifies the data-pipeline code because we can
pass around one strongly-typed instance instead of half a dozen plain
strings.
"""

from unitok import UniTok


class ColumnMap:
    """
    Container for dataset column identifiers and their corresponding
    vocabulary names.

    Parameters
    ----------
    history_col : str
        Column that stores a sequence of historical interactions
        (e.g. click history).
    item_col : str
        Column that stores the *current* item id.
    label_col : str
        Column that stores the binary / categorical target label.
    user_col : str
        Column that stores the user id.
    group_col : str
        Column that groups multiple rows into a single *session* or user
        (defaults to the same value as `user_col`).
    neg_col : str | None
        Optional column that contains negative samples for each row.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        history_col: str = "history",
        item_col: str = "item_id",
        label_col: str = "click",
        user_col: str = "user_id",
        group_col: str = "user_id",
        neg_col: str | None = None,
    ) -> None:
        # raw column names ------------------------------------------------
        self.history_col = history_col
        self.item_col = item_col
        self.label_col = label_col
        self.group_col = group_col
        self.user_col = user_col
        self.neg_col = neg_col

        # an artificial column sometimes added to represent padding masks
        self.mask_col = "__clicks_mask__"

        # corresponding vocabulary names (to be filled in later) ---------
        self.history_vocab: str | None = None
        self.item_vocab: str | None = None
        self.label_vocab: str | None = None
        self.user_vocab: str | None = None
        self.group_vocab: str | None = None

    # ------------------------------------------------------------------ #
    # Initialisation helper                                              #
    # ------------------------------------------------------------------ #
    def set_column_vocab(self, inter_ut: UniTok) -> None:
        """
        Populate the `*_vocab` attributes by querying a *fitted* UniTok
        instance.

        UniTok keeps per-column tokenisers whose vocabularies are named
        according to the dataset’s meta-information.  This method
        extracts the names so that downstream components can refer to
        vocabularies without having to carry the complete UniTok object
        around.

        Parameters
        ----------
        inter_ut : unitok.UniTok
            A fully initialized / fitted UniTok object for the dataset.
        """

        def col_to_vocab(col: str) -> str:
            """
            Helper that maps a column name to the underlying vocabulary
            identifier via UniTok’s meta information.
            """
            return inter_ut.meta.features[col].tokenizer.vocab.name

        # fetch vocab names for every relevant column --------------------
        self.history_vocab = col_to_vocab(self.history_col)
        self.item_vocab = col_to_vocab(self.item_col)
        self.label_vocab = col_to_vocab(self.label_col)
        self.user_vocab = col_to_vocab(self.user_col)
        self.group_vocab = col_to_vocab(self.group_col)
