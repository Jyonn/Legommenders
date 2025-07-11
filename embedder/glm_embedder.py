"""
glm_embedder.py

Adapter that exposes the **ChatGLM** family of language models through
the generic `BaseEmbedder` interface.

Why another adapter?
--------------------
The official ChatGLM checkpoints hosted on HuggingFace are shipped as
`AutoModelForCausalLM` objects whose *actual* transformer backbone is
nested in the attribute `.transformer`.  In addition, GLM keeps an
extra set of *special tokens* at the tail of the embedding matrix
(`<sop>`, `<eop>`, padding, …) which should typically be **ignored**
when exporting the *ordinary* vocabulary embeddings.

The class hierarchy therefore looks like

    BaseEmbedder  <-  GLMEmbedder  <-  GLM4TH9BEmbedder

Only `GLMEmbedder` contains logic; concrete subclasses (one per
checkpoint) simply inherit it unchanged.

Key steps implemented here
--------------------------
1) Resolve checkpoint / cache directory through `ModelInit.get(<id>)`
   where `<id>` is produced by `BaseEmbedder.__str__`
   (e.g. "glm4th9b").

2) Load the model via
       AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
   The `trust_remote_code` flag is **required** because ChatGLM has its
   own custom modelling code.

3) Retrieve the `vocab_size` from the corresponding tokenizer to slice
   out the *usable* part of the embedding matrix.

4) Wrap the sliced tensor in a fresh `nn.Embedding` so that
   `BaseEmbedder.get_embeddings()` can work as usual.
"""

from __future__ import annotations

import abc

from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from embedder.base_embedder import BaseEmbedder
from model.common.glm_interface import ChatGLMModel
from utils.config_init import ModelInit


class GLMEmbedder(BaseEmbedder, abc.ABC):
    """
    Base class for every ChatGLM checkpoint.

    Child classes (e.g. `GLM4TH9BEmbedder`) do not need any additional
    code—the only difference is the **identifier** returned by
    `__str__`, which in turn selects the proper entry in `ModelInit`.
    """

    # Annotate for IDEs / type checkers; actual assignment happens in __init__
    transformer: ChatGLMModel

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        super().__init__()

        # Resolve the model name / path from configuration
        model_name: str = ModelInit.get(str(self))

        # 1) Load the full causal-LM wrapper ----------------------------
        glm_lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True  # mandatory for 3rd-party modelling code
        )

        # 2) Keep only the underlying transformer (saves memory) --------
        #    transformers' ChatGLM architecture stores the main network
        #    in the attribute `.transformer`.
        self.transformer = glm_lm.transformer  # type: ignore[attr-defined]

        # 3) Tokenizer is needed to obtain *vocab_size* -----------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    # ------------------------------------------------------------------ #
    # BaseEmbedder abstract method implementation                         #
    # ------------------------------------------------------------------ #
    def _get_embeddings(self) -> nn.Embedding:
        """
        Return a *pruned* `nn.Embedding` object that contains only the
        first `vocab_size` rows of the original embedding matrix.

        ChatGLM appends special tokens after the normal vocabulary, so
        we slice them away before wrapping the tensor.
        """
        vocab_size: int = self.tokenizer.vocab_size

        # Original embedding layer
        base_embeddings = self.transformer.get_input_embeddings().weight  # Tensor

        # Slice to keep exactly `vocab_size` entries
        trimmed_weights = base_embeddings[:vocab_size]

        # Wrap into a fresh Embedding layer so downstream code can assume
        # a *stand-alone* `nn.Embedding` (no gradient / tied weights).
        return nn.Embedding.from_pretrained(trimmed_weights)


# ---------------------------------------------------------------------- #
# Concrete checkpoints                                                   #
# ---------------------------------------------------------------------- #
class GLM4TH9BEmbedder(GLMEmbedder):
    """
    ChatGLM-4TH-9B (≈9 billion parameters) embedding extractor.
    """
    pass
