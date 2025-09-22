"""
recbench_processor.py

A generic processor for datasets released in the **RecBench** benchmark
(http://recbench.github.io).  RecBench provides pre-processed parquet files that
already contain:

    • items.parquet      (item meta-data)
    • users.parquet      (user histories + negatives)
    • finetune.parquet   (finetune / validation interactions)
    • test.parquet       (inference interactions)
    • valid_user_set_*.txt (list of user-ids reserved for validation)

Consequently, this processor mainly needs to

1. Read the parquet files,
2. Build a TRAIN / VALID / TEST split according to `valid_user_set`,
3. Add various tokenisers to `UniTok`,
4. Produce a YAML snippet that can be consumed by a generic *trainer*
   (see `generate_data_configuration`).

Concrete RecBench domains (Automotive, Beauty, …) inherit from
`RecBenchProcessor` and only customise column names (`UID_COL`, …),
prompt prefix (`PROMPT`) and textual attributes (`attrs`).
"""
import abc
import os.path

import pandas as pd
from unitok import BertTokenizer, TransformersTokenizer, GloVeTokenizer

from embedder.glove_embedder import GloVeEmbedder
from processor.base_processor import BaseProcessor, Interactions
from utils import io
from utils.config_init import ModelInit


class RecBenchProcessor(BaseProcessor, abc.ABC):
    """
    Base class reused by all individual RecBench *domains*.
    """

    # --------------------- dataset-specific constants -----------------
    PROMPT: str                           # natural language prefix
    NEG_COL: str = "neg"                  # column with negative items
    NEG_TRUNCATE = 100                    # max negatives per user

    BASE_STORE_DIR = os.path.join("data", "recbench")

    # =================================================================
    #                           Construction
    # =================================================================
    def __init__(self, data_dir: str):
        """
        Parameters
        ----------
        data_dir : str
            Directory that already contains the parquet files generated
            by the official RecBench scripts.
        """
        super().__init__(data_dir=data_dir)

        # ------------------------------------------------------------------
        # Load parquet files created by the RecBench toolkit
        # ------------------------------------------------------------------
        self.item_df     = pd.read_parquet(os.path.join(data_dir, "items.parquet"))
        self.user_df     = pd.read_parquet(os.path.join(data_dir, "users.parquet"))
        self.finetune_df = pd.read_parquet(os.path.join(data_dir, "finetune.parquet"))
        self.test_df     = pd.read_parquet(os.path.join(data_dir, "test.parquet"))

        # `history` is stored as numpy array → convert to native list
        self.user_df[self.HIS_COL] = self.user_df[self.HIS_COL].apply(
            lambda x: x if isinstance(x, list) else x.tolist()
        )

        # --------------------------------------------------------------
        # Build TRAIN / VALID split according to pre-defined user ids
        # --------------------------------------------------------------
        self.valid_user_set = self.load_valid_user_set(valid_ratio=0.1)

        self.train_df = self.finetune_df[
            ~self.finetune_df[self.UID_COL].isin(self.valid_user_set)
        ]
        self.valid_df = self.finetune_df[
            self.finetune_df[self.UID_COL].isin(self.valid_user_set)
        ]

    # =================================================================
    #                   UniTok tokeniser configuration
    # =================================================================
    def config_item_tokenization(self):
        """
        Attach three tokenisers (BERT, LLaMA-1, GloVe) so downstream code
        can flexibly pick *any* LM as `${lm}` placeholder.
        """
        bert_tokenizer  = BertTokenizer(vocab="bert")
        llama1_tokenizer = TransformersTokenizer(
            vocab="llama1", key=ModelInit.get("llama1")
        )
        glove_tokenizer = GloVeTokenizer(vocab=GloVeEmbedder.get_glove_vocab())

        # Register them as additional item tokenisers
        self.add_item_tokenizer(bert_tokenizer)
        self.add_item_tokenizer(llama1_tokenizer)
        self.add_item_tokenizer(glove_tokenizer)

    # =================================================================
    #                           Helper I/O
    # =================================================================
    def load_valid_user_set(self, valid_ratio: float) -> set:
        """
        Read the file `valid_user_set_<ratio>.txt` that contains one user
        id per line and return it as a Python `set`.
        """
        # with open(
        #     os.path.join(self.data_dir, f"valid_user_set_{valid_ratio}.txt"), "r"
        # ) as f:
        #     return {line.strip() for line in f}
        lines = io.file_load(os.path.join(self.data_dir, f"valid_user_set_{valid_ratio}.txt")).split('\n')
        return {line.strip() for line in lines}

    # ----------------------------- loaders ---------------------------
    def load_items(self) -> pd.DataFrame:
        """
        Fill missing textual attributes with “[empty]”, generate prompt
        prefixes and return the DataFrame as-is.
        """
        # Replace NaN with a special token so that tokenisers see a value
        for attr in self.attrs:
            self.item_df[attr] = self.item_df[attr].fillna("[empty]")

        # Add static prompt columns used by LLM input construction
        self.item_df["prompt"] = self.PROMPT
        for attr in self.attrs:
            self.item_df[f"prompt_{attr}"] = attr.upper()[0] + attr[1:].lower() + ": "
        return self.item_df

    def load_users(self) -> pd.DataFrame:
        """Just return the pre-processed users.parquet DataFrame."""
        return self.user_df

    def load_interactions(self) -> Interactions:
        """
        BUILD:
            • TRAIN  = interactions of users *not* in valid_user_set
            • VALID  = interactions of users in valid_user_set
            • TEST   = provided benchmark test set
        """
        return Interactions(self.train_df, self.valid_df, self.test_df)

    # =================================================================
    #                     YAML data-config generator
    # =================================================================
    def generate_data_configuration(self):
        """
        Produce a YAML dict that encapsulates paths & column mapping so
        the *trainer* script can stay fully generic.
        """
        return dict(
            name=self.get_name(),
            item=dict(
                ut=self.item_save_dir,
                inputs=[attr + "@${lm}" for attr in self.attrs],  # e.g. "title@bert"
            ),
            user=dict(
                ut=self.user_save_dir,
                truncate=50,
            ),
            inter=dict(
                train=self.get_save_dir(Interactions.train),
                dev=self.get_save_dir(Interactions.valid),
                test=self.get_save_dir(Interactions.test),
                filters=dict(history=["lambda x: x"]),
            ),
            column_map=dict(
                item_col=self.IID_FEAT,
                user_col=self.UID_FEAT,
                history_col=self.HIS_FEAT,
                neg_col=self.NEG_COL,
                label_col=self.LBL_FEAT,
                group_col=self.UID_FEAT,
            ),
        )

    # =================================================================
    #                Overridden load to emit YAML when needed
    # =================================================================
    def load(self, regenerate: bool = False):
        """
        Calls the parent `load` and additionally writes `config/data/*.yaml`
        if it doesn’t already exist (or if regenerate=True).
        """
        super().load(regenerate=regenerate)

        yaml_path = os.path.join("config", "data", f"{self.get_name()}.yaml")

        # Create data-config file once; warn user where to find it
        if not os.path.exists(yaml_path) or regenerate:
            data_config = self.generate_data_configuration()
            io.yaml_save(data_config, yaml_path)

            print(
                f"Data configuration saved to {yaml_path}, "
                "please use `python trainer.py --data {yaml_path} --lm ...` to train"
            )
