"""
opt_embedder.py

Adapters for Meta-AI’s OPT (decoder-only) language-model family that
expose their token-embedding tables through the project-wide
`BaseEmbedder` interface.

Hierarchy
---------
BaseEmbedder        # see embedder/base_embedder.py
└── OPTEmbedder     # contains the shared implementation
    ├── OPTBaseEmbedder   # OPT-125M / OPT-350M … whatever “base” points to
    └── OPTLargeEmbedder  # larger checkpoint (e.g. OPT-1.3B / 2.7B …)

Only `OPTEmbedder` carries executable logic.  
The concrete subclasses differ *solely* by the textual identifier
returned from `str(self)` which is subsequently passed to
`ModelInit.get()` to retrieve the checkpoint name or path.

Key implementation notes
------------------------
1)  The constructor loads
        a) the transformer weights via `AutoModel.from_pretrained`
           *returned object is of type `OPTModel`*,
        b) the matching tokenizer (needed to know `vocab_size`).

2)  OPT’s embedding matrix is stored in
        `transformer.decoder.embed_tokens`
    which includes *extra* rows for special tokens reserved by
    HuggingFace.  We slice the matrix up to `tokenizer.vocab_size`
    to obtain a clean  embedding table and wrap it in a fresh,
    stand-alone `nn.Embedding` layer.

3)  The returned layer is **unfrozen** by default so callers can decide
    whether to keep the weights static or fine-tune them further.
"""

from __future__ import annotations

import abc
from typing import cast

from torch import nn
from transformers import AutoModel, AutoTokenizer, OPTModel

from embedder.base_embedder import BaseEmbedder
from utils.config_init import ModelInit


class OPTEmbedder(BaseEmbedder, abc.ABC):
    """
    Common logic for every OPT checkpoint.

    Subclasses simply specialise the *identifier* picked up by
    `BaseEmbedder.__str__` (“optbase”, “optlarge”, …).
    """

    # Keep explicit attribute for better IDE support / type checking
    transformer: OPTModel

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        super().__init__()

        # Resolve checkpoint path / Hub name via configuration helper
        model_name: str = ModelInit.get(str(self))

        # Load model weights (AutoModel returns the correct subclass)
        self.transformer = cast(
            OPTModel,
            AutoModel.from_pretrained(model_name)
        )

        # Tokenizer is required to know the effective vocabulary size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ------------------------------------------------------------------ #
    # BaseEmbedder: abstract method implementation                        #
    # ------------------------------------------------------------------ #
    def _get_embeddings(self) -> nn.Embedding:
        """
        Extract the usable part of OPT’s shared embedding matrix and
        wrap it in a new `nn.Embedding` module.

        Returns
        -------
        torch.nn.Embedding
            Shape = (vocab_size, hidden_dim)
        """
        vocab_size: int = self.tokenizer.vocab_size

        # Original embedding weights (Tensor)
        full_weights = self.transformer.decoder.embed_tokens.weight

        # Slice away rows > vocab_size (special reserved tokens)
        trimmed_weights = full_weights[:vocab_size]

        # Wrap into a stand-alone Embedding layer (trainable by default)
        return nn.Embedding.from_pretrained(trimmed_weights, freeze=False)


# ---------------------------------------------------------------------- #
# Concrete checkpoints                                                   #
# ---------------------------------------------------------------------- #
class OPTBaseEmbedder(OPTEmbedder):
    """Embedding extractor for the *base-sized* OPT checkpoint."""
    pass


class OPTLargeEmbedder(OPTEmbedder):
    """Embedding extractor for the *large* OPT checkpoint."""
    pass
