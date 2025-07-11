"""
llama_embedder.py

Adapters for the LLaMA family of decoder-only language models that make
their **token embedding tables** accessible through the common
`BaseEmbedder` API.

Class hierarchy
---------------
BaseEmbedder           # generic helper (see embedder/base_embedder.py)
└── LlamaEmbedder      # shared logic for every LLaMA checkpoint
    ├── Llama1Embedder # Meta-AI LLaMA-1
    ├── Llama2Embedder # Meta-AI LLaMA-2
    └── Llama3Embedder # Meta-AI LLaMA-3

Only `LlamaEmbedder` contains real code; the concrete subclasses differ
solely by the **identifier** returned from `str(self)` which is later
used to look up the checkpoint path via `ModelInit.get`.

Implementation details
----------------------
1) The constructor loads the full `AutoModelForCausalLM` wrapper (from
   HuggingFace) and stores the underlying `LlamaForCausalLM` instance in
   `self.transformer` so that it can be checkpointed or inspected later.

2) `_get_embeddings()` returns the *shared* token-embedding layer
   (`transformer.model.embed_tokens`) required by
   `BaseEmbedder.get_embeddings()`.
"""

from __future__ import annotations

import abc
from typing import cast

from transformers import AutoModelForCausalLM, LlamaForCausalLM

from embedder.base_embedder import BaseEmbedder
from utils.config_init import ModelInit


class LlamaEmbedder(BaseEmbedder, abc.ABC):
    """
    Common functionality for all LLaMA-based embedders.

    Subclasses *only* select a different checkpoint via their class name
    (see `BaseEmbedder.__str__`).
    """

    # Annotate for IDE / static type checkers
    transformer: LlamaForCausalLM

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        super().__init__()

        # Resolve checkpoint name / path from configuration
        model_name: str = ModelInit.get(str(self))

        # Load model weights; `AutoModelForCausalLM` returns a wrapper
        # whose `.model` attribute holds the actual transformer
        # (LlamaForCausalLM).  We keep the specialised type for clarity.
        self.transformer = cast(
            LlamaForCausalLM,
            AutoModelForCausalLM.from_pretrained(model_name)
        )

    # ------------------------------------------------------------------ #
    # BaseEmbedder abstract method implementation                         #
    # ------------------------------------------------------------------ #
    def _get_embeddings(self):
        """
        Return the `nn.Embedding` layer that stores the vocabulary lookup
        weights for LLaMA.
        """
        return self.transformer.model.embed_tokens


# ---------------------------------------------------------------------- #
# Concrete checkpoints (no additional code needed)                       #
# ---------------------------------------------------------------------- #
class Llama1Embedder(LlamaEmbedder):
    """Embedding extractor for Meta-AI LLaMA-1."""
    pass


class Llama2Embedder(LlamaEmbedder):
    """Embedding extractor for Meta-AI LLaMA-2."""
    pass


class Llama3Embedder(LlamaEmbedder):
    """Embedding extractor for Meta-AI LLaMA-3."""
    pass
