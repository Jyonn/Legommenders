"""
embedding_hub.py

Management layer that unifies *all* embedding look-ups used inside the
project.  It solves three practical problems:

1) Book-keeping
   ----------------
   Different parts of the code might request embeddings for the same
   vocabulary / feature multiple times.  `EmbeddingHub` keeps one shared
   `nn.Embedding` (or transformation wrapper) per vocabulary and returns
   references instead of creating duplicates.

2) Pre-trained vectors
   --------------------
   Vocabularies or columns can be initialised from a `.npy` file that
   stores pre-trained weights.  The user can decide whether those weights
   should be kept *frozen* or be fine-tuned (`frozen=True / False`).
   Optionally a projection layer (`Transformation`) is inserted so the
   external dimensionality does not have to match the project-wide
   `embedding_dim`.

3) Dimension alignment
   --------------------
   With multiple pre-trained sources it is almost guaranteed that vector
   sizes will differ.  
   `Transformation` is a small `nn.Module` that applies
       1. embedding look-up,
       2. linear projection,
       3. dropout
   so that every embedding ultimately delivers tensors of shape
   `[*, embedding_dim]`.

Terminology
-----------
• “vocab embedding”   – keyed by *vocabulary name* (global across all
  datasets).

• “feature embedding” – keyed by *column / feature name* in a specific
  dataset (might re-use an existing vocabulary).

Both live in their own `ModuleDict`, namely `vocab_table` and
`feature_table`.

Supported transformation policies
---------------------------------
LINEAR : always insert a linear projection regardless of the original
         vector size.
AUTO   : only insert the projection if the incoming size differs from
         the requested `embedding_dim`.

The *global* policy is chosen when the hub is instantiated but can be
overridden for individual pre-trained files.
"""

from __future__ import annotations

from typing import Dict, Iterable, cast

import numpy as np
import torch
from torch import nn
from unitok import Feature, Vocab
from pigmento import pnt

from loader.env import Env
from loader.ut.lego_ut import LegoUT


# --------------------------------------------------------------------- #
# Helper modules                                                        #
# --------------------------------------------------------------------- #
class Transformation(nn.Module):
    """
    Wrapper that takes an `nn.Embedding` whose **output dimension may
    differ** from the desired one and brings it in line via

        y = Dropout( Linear( Embedding(x) ) )

    This makes it possible to mix vectors of different sizes inside the
    same model with minimal boilerplate.
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        to_dimension: int,
        transformation_dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.linear = nn.Linear(embedding.weight.data.shape[1], to_dimension)
        self.dropout = nn.Dropout(transformation_dropout)

    def forward(self, indexes: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportMissingTypeStubs]
        return self.dropout(self.linear(self.embedding(indexes)))


class PretrainedEmbedding:
    """
    Plain data-object that holds the ingredients required to turn a
    pre-trained weight matrix into a ready-to-use model component.
    """

    def __init__(
        self,
        embedder: nn.Embedding,
        transformation: str,
        transformation_dropout: float,
        frozen: bool,
    ) -> None:
        self.embedder = embedder
        self.transformation = transformation
        self.transformation_dropout = transformation_dropout
        self.frozen = frozen


# --------------------------------------------------------------------- #
# Main registry                                                         #
# --------------------------------------------------------------------- #
class EmbeddingHub:
    LINEAR = "linear"
    AUTO = "auto"
    DEFAULT = "default"  # “inherit from global setting”

    # valid values for the respective contexts
    global_types = {LINEAR, AUTO}
    pretrained_types = {DEFAULT, LINEAR, AUTO}

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        embedding_dim: int,
        transformation: str,
        transformation_dropout: float,
    ) -> None:
        if transformation not in self.global_types:
            raise ValueError(
                f"invalid transformation type {transformation}, "
                f"expected {self.global_types}"
            )

        pnt(f"global transformation type: {transformation}")
        pnt(f"global transformation dropout: {transformation_dropout}")

        self.embedding_dim = embedding_dim
        self.transformation = transformation
        self.transformation_dropout = transformation_dropout

        # book-keeping ---------------------------------------------------
        self._vocab_size: Dict[str, int] = {}  # sanity-check helper
        self.vocab_table = nn.ModuleDict()
        self.feature_table = nn.ModuleDict()

        self._pretrained_vocab_embeddings: Dict[str, PretrainedEmbedding] = {}
        self._pretrained_feature_embeddings: Dict[str, PretrainedEmbedding] = {}

    # ------------------------------------------------------------------ #
    # Loading of external vectors                                        #
    # ------------------------------------------------------------------ #
    def load_pretrained_embedding(
        self,
        path: str,
        *,
        vocab_name: str | None = None,
        col_name: str | None = None,
        transformation: str = DEFAULT,
        transformation_dropout: float | None = None,
        frozen: bool = True,
    ) -> None:
        """
        Register a `.npy` file (`path`) that contains a complete embedding
        matrix.  The matrix must match the size of the vocabulary /
        feature it is going to be attached to.

        Only *metadata* is stored at this point; the actual `nn.Embedding`
        module is instantiated eagerly because we need its `.weight` shape
        to perform sanity-checks.
        """
        # Validate mutually exclusive parameters ------------------------
        if vocab_name is None and col_name is None:
            raise ValueError("vocab_name or col_name must be specified")
        if vocab_name is not None and col_name is not None:
            raise ValueError("only one of vocab_name and col_name can be specified")

        # Read the weight matrix from disk ------------------------------
        name = cast(str, vocab_name or col_name)
        embedding_arr = np.load(path)
        embedding_tensor = torch.tensor(embedding_arr, dtype=torch.float32)
        embedding = nn.Embedding.from_pretrained(embedding_tensor)

        pnt(f"load pretrained embedding {name} of shape {embedding.weight.shape}:")

        # Parameter checks / defaults -----------------------------------
        if name == "<vocab_name>":  # template placeholder left untouched
            raise ValueError(
                "please specify the vocab name for the pretrained embedding in the config"
            )

        if transformation not in self.pretrained_types:
            raise ValueError(
                f"invalid transformation type {transformation}, "
                f"expected {self.pretrained_types}"
            )
        if transformation == self.DEFAULT:
            pnt("--- use default transformation")
            transformation = self.transformation

        if transformation_dropout is None:
            pnt("--- use default transformation dropout")
            transformation_dropout = self.transformation_dropout

        pnt(f"--- pretrained transformation type: {transformation}")
        pnt(f"--- pretrained transformation dropout: {transformation_dropout}")

        # Store everything in the appropriate directory -----------------
        target = (
            self._pretrained_vocab_embeddings
            if vocab_name is not None
            else self._pretrained_feature_embeddings
        )
        pnt(
            "--- saved in "
            + ("pretrained vocab embeddings" if vocab_name else "pretrained feature embeddings")
        )

        target[name] = PretrainedEmbedding(
            embedder=embedding,
            transformation=transformation,
            transformation_dropout=transformation_dropout,
            frozen=frozen,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _process_pretrained_embedding(
        self, name: str, size: int, pe: PretrainedEmbedding
    ) -> None:
        """
        Finalise a `PretrainedEmbedding` so that it can be inserted into a
        `ModuleDict`:

        • check size consistency
        • set `requires_grad` according to `pe.frozen`
        • optionally wrap in `Transformation`
        """
        frozen_str = "frozen" if pe.frozen else "unfrozen"
        pnt(f"--- load {frozen_str} vocab: {name} {pe.embedder.weight.shape}")

        # ----------------------------------------------------------------
        # 1) Size check
        # ----------------------------------------------------------------
        if int(pe.embedder.weight.shape[0]) != size:
            raise ValueError(f"{name} does not match the expected vocab size {size}")

        # ----------------------------------------------------------------
        # 2) Make weights trainable or not
        # ----------------------------------------------------------------
        pe.embedder.weight.requires_grad = not pe.frozen

        # ----------------------------------------------------------------
        # 3) Insert projection layer if necessary
        # ----------------------------------------------------------------
        embedding_size = int(pe.embedder.weight.data.shape[1])

        need_projection = (
            embedding_size != self.embedding_dim or self.transformation == self.LINEAR
        )
        if need_projection:
            pnt(
                f"--- transform {name} embedding from size "
                f"{embedding_size} to {self.embedding_dim}"
            )
            pe.embedder = Transformation(
                embedding=cast(nn.Embedding, pe.embedder),
                to_dimension=self.embedding_dim,
                transformation_dropout=pe.transformation_dropout,
            )

    # ------------------------------------------------------------------ #
    # Public build / register helpers                                    #
    # ------------------------------------------------------------------ #
    def build_feature_embedding(self, feature: Feature) -> bool:
        """
        Create (if necessary) an embedding for *feature* and add it to
        `self.feature_table`.

        Returns
        -------
        bool
            True if a new module was added, False if it already existed.
        """
        if feature.name in self.feature_table:
            return False  # already built

        if feature.name not in self._pretrained_feature_embeddings:
            # No pre-trained weights → nothing to do (feature will fall
            # back to its vocabulary embedding)
            return False

        pnt(
            f"build pretrained embedding for feature {feature.name} "
            f"({feature.tokenizer.vocab.size}, {self.embedding_dim})"
        )

        pe = self._pretrained_feature_embeddings[feature.name]
        self._process_pretrained_embedding(
            feature.name, feature.tokenizer.vocab.size, pe
        )

        # Move to correct device and register
        self.feature_table.add_module(feature.name, pe.embedder.to(Env.device))
        return True

    def build_vocab_embedding(self, vocab: Vocab) -> None:
        """
        Ensure that an embedding for *vocab* exists in `self.vocab_table`.
        """
        if vocab.name in self.vocab_table:
            return  # already present

        if vocab.name not in self._pretrained_vocab_embeddings:
            # Fresh randomly initialized embedding
            pnt(f"create vocab {vocab.name} ({vocab.size}, {self.embedding_dim})")
            self.vocab_table.add_module(
                vocab.name,
                nn.Embedding(
                    num_embeddings=vocab.size,
                    embedding_dim=self.embedding_dim,
                ).to(Env.device),
            )
            return

        # Otherwise: build from pre-trained weights
        pnt(f"build pretrained embedding for vocab {vocab.name} ({vocab.size}, {self.embedding_dim})")
        pe = self._pretrained_vocab_embeddings[vocab.name]
        self._process_pretrained_embedding(vocab.name, vocab.size, pe)
        self.vocab_table.add_module(vocab.name, pe.embedder.to(Env.device))

    # ------------------------------------------------------------------ #
    # Registration helpers (called by data pipeline)                     #
    # ------------------------------------------------------------------ #
    def register_vocab(self, vocab: Vocab) -> None:
        """
        Remember the expected `vocab.size` (for conflict detection) and
        ensure an embedding exists.
        """
        if vocab.name in self._vocab_size:
            if self._vocab_size[vocab.name] != vocab.size:
                raise ValueError(
                    f"conflict in vocab {vocab.name}: "
                    f"{self._vocab_size[vocab.name]} vs {vocab.size}"
                )
            return

        self._vocab_size[vocab.name] = vocab.size
        self.build_vocab_embedding(vocab)

    def register_ut(self, ut: LegoUT, used_cols: Iterable[str]) -> None:
        """
        Inspect a `LegoUT` object and build embeddings for every column
        listed in `used_cols`.
        """
        for col in used_cols:
            feature = ut.meta.features[col]
            self.build_feature_embedding(feature)

            # Make sure the underlying vocabulary is covered as well
            vocab = feature.tokenizer.vocab
            self.register_vocab(vocab)

    # ------------------------------------------------------------------ #
    # Forward access                                                     #
    # ------------------------------------------------------------------ #
    def __call__(self, vocab_name: str, col_name: str | None = None) -> nn.Module:
        """
        Retrieve an embedding module.  Column-specific embeddings take
        precedence; if none exists we fall back to the vocabulary table.
        """
        if col_name and col_name in self.feature_table:
            return self.feature_table[col_name]
        return self.vocab_table[vocab_name]
