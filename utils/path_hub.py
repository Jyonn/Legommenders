"""
path_hub.py

Centralized *path manager* used throughout the project to keep file and
directory names consistent.

Given a dataset name, a model name and a unique experiment `signature`,
the class provides convenient properties pointing to:

    • checkpoint_base_dir   checkpoints/<data>/<model>/
    • log_path              <base>/<signature>.log
    • cfg_path              <base>/<signature>.json
    • ckpt_path             <base>/<signature>.pt
    • result_path           <base>/<signature>.csv

The constructor will automatically:

    1. Create the base checkpoint directory if it does not exist.
    2. Touch an empty *.log* file so that downstream code can simply
       append to it without worrying about “file not found” errors.
"""

import os

from utils import io


class PathHub:
    """
    Parameters
    ----------
    data_name   : str
        Identifier of the dataset, e.g. "ml-1m".
    model_name  : str
        Identifier of the model / algorithm, e.g. "DIN".
    signature   : str
        Short hash or experiment tag that disambiguates runs with the
        same data+model but different hyper-parameters.
    """

    def __init__(self, data_name: str, model_name: str, signature: str):
        self.data_name = data_name
        self.model_name = model_name
        self.signature = signature

        # -----------------------------------------------------------------
        # Ensure the base directory exists
        # -----------------------------------------------------------------
        os.makedirs(self.checkpoint_base_dir, exist_ok=True)

        # -----------------------------------------------------------------
        # “Touch” the log file (create an empty file if missing)
        # -----------------------------------------------------------------
        # with open(self.log_path, "w") as f:
        #     pass
        io.file_save(self.log_path, '')

    # ---------------------------------------------------------------------
    # Path helpers (lazy properties)
    # ---------------------------------------------------------------------
    @property
    def checkpoint_base_dir(self) -> str:
        """
        Root directory that stores *all* artefacts for the given
        (data, model) pair.

            checkpoints/<data_name>/<model_name>/
        """
        return os.path.join("checkpoints", self.data_name, self.model_name)

    @property
    def log_path(self) -> str:
        """
        Path to the textual log file of the current experiment run.
        """
        return os.path.join(self.checkpoint_base_dir, f"{self.signature}.log")

    @property
    def cfg_path(self) -> str:
        """
        Path where the JSON serialized configuration will be saved.
        """
        return os.path.join(self.checkpoint_base_dir, f"{self.signature}.json")

    @property
    def ckpt_path(self) -> str:
        """
        Path to the PyTorch checkpoint (*.pt) file.
        """
        return os.path.join(self.checkpoint_base_dir, f"{self.signature}.pt")

    @property
    def result_path(self) -> str:
        """
        Path where tabular evaluation results (CSV) will be stored.
        """
        return os.path.join(self.checkpoint_base_dir, f"{self.signature}.csv")
