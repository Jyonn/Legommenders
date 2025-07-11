"""
glove_embedder.py

Utility that turns the *pre-trained* GloVe vectors
(https://nlp.stanford.edu/projects/glove/) into a `torch.nn.Embedding`
layer so it can be consumed by the rest of the code base through the
generic `BaseEmbedder` interface.

Unlike the other embedders in this project, GloVe is a **static**
embedding – its parameters are *frozen* and there is no underlying
`self.transformer` model.

Workflow implemented by `_get_embeddings()`:
    1.  Download the 6B GloVe archive (≈ 862 MB) if it is not present
        locally.
    2.  Unzip the required file (`glove.6B.300d.txt`).
    3.  Read every line → split into <word> <300-dim float vector>.
    4.  Populate a `unitok.Vocab` instance with the words **in order**
        so that indices match the embedding rows.
    5.  Convert the collected list of vectors into a `torch.Tensor`.
    6.  Wrap the tensor in an **immutable** `nn.Embedding` created with
        `freeze=True`.
    7.  Persist the vocabulary to disk next to the embedding file so
        downstream code can reuse it without re-parsing the text file.

`get_glove_vocab()` is a convenience helper that loads the previously
saved vocabulary or raises a descriptive error if it has not been
generated yet.
"""

from __future__ import annotations

import os
import subprocess
import zipfile
from typing import List

import torch
from pigmento import pnt
from torch import nn
from unitok import Vocab

from embedder.base_embedder import BaseEmbedder


class GloVeEmbedder(BaseEmbedder):
    """
    Converts the 300-d GloVe-6B vectors into a frozen `nn.Embedding`.
    """

    # ------------------------------------------------------------------ #
    # Constants describing where and what to download                     #
    # ------------------------------------------------------------------ #
    DL_URI = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"

    FILE_DIR = os.path.join("data", "embeddings")          # destination root
    ZIP_NAME = "glove.6B.zip"
    ZIP_PATH = os.path.join(FILE_DIR, ZIP_NAME)

    FILE_NAME = "glove.6B.300d.txt"                        # inside the zip
    FILE_PATH = os.path.join(FILE_DIR, FILE_NAME)

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        super().__init__()

        # Ensure <data/embeddings/> exists beforehand
        os.makedirs(self.FILE_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # BaseEmbedder abstract method implementation                         #
    # ------------------------------------------------------------------ #
    def _get_embeddings(self) -> nn.Embedding:
        """
        Return a frozen `nn.Embedding` loaded with the pre-trained GloVe
        300-dimensional vectors.
        """
        # -------------------------------------------------------------- #
        # 1. Download archive if necessary                               #
        # -------------------------------------------------------------- #
        if not os.path.exists(self.ZIP_PATH):
            pnt(f"🔽  Downloading GloVe embeddings from {self.DL_URI} …")
            subprocess.run(                       # nosec B603, B607
                ["curl", "-L", "-o", self.ZIP_PATH, self.DL_URI],
                check=True,
            )
            pnt(f"   ✓  Archive saved to {self.ZIP_PATH}")

        # -------------------------------------------------------------- #
        # 2. Unzip the required txt file                                 #
        # -------------------------------------------------------------- #
        if not os.path.exists(self.FILE_PATH):
            pnt("📦  Extracting glove.6B.300d.txt …")
            with zipfile.ZipFile(self.ZIP_PATH, "r") as zip_ref:
                zip_ref.extract(self.FILE_NAME, path=self.FILE_DIR)
            pnt(f"   ✓  Unzipped to {self.FILE_DIR}")

        # -------------------------------------------------------------- #
        # 3. Parse vectors & build vocab                                 #
        # -------------------------------------------------------------- #
        vocab = Vocab(name="glove")
        embeddings: List[List[float]] = []

        with open(self.FILE_PATH, encoding="utf-8") as fh:
            for line in fh:
                word, *values = line.rstrip().split()
                vocab.append(word)                               # keep order!
                embeddings.append([float(x) for x in values])

        # Convert to tensor  (len(vocab) × 300)
        weight_matrix = torch.tensor(embeddings)

        # Persist vocabulary next to embedding for future reuse ---------
        vocab.save(self.FILE_DIR)
        pnt(
            f"💾  Saved vocabulary to {vocab.filepath(self.FILE_DIR)} "
            f"({len(vocab)} tokens)"
        )

        # -------------------------------------------------------------- #
        # 4. Wrap in a *frozen* nn.Embedding and return                  #
        # -------------------------------------------------------------- #
        return nn.Embedding.from_pretrained(weight_matrix, freeze=True)

    # ------------------------------------------------------------------ #
    # Public helper                                                      #
    # ------------------------------------------------------------------ #
    @classmethod
    def get_glove_vocab(cls) -> Vocab:
        """
        Load the GloVe vocabulary generated by an earlier run.

        Raises
        ------
        FileNotFoundError
            If the vocabulary has not been built yet.  In that case the
            user should run the normal embedding extraction first.
        """
        vocab = Vocab(name="glove")
        vocab_path = vocab.filepath(cls.FILE_DIR)

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"GloVe vocabulary not found at '{vocab_path}'. "
                "Please run the embedder once (e.g. "
                "`python embed.py --model glove`) to create it."
            )

        return vocab.load(cls.FILE_DIR)
