"""
bert_embedder.py

Wrappers around HuggingFace's `BertModel` that comply with the
`BaseEmbedder` interface defined in *embedder/base_embedder.py*.

Two thin subclasses are provided – `BertBaseEmbedder` and
`BertLargeEmbedder` – which differ only in the **checkpoint name**
returned by `ModelInit`.  The actual logic for loading BERT and
exporting its word-piece embedding table lives in the
`BertEmbedder` mix-in.

Design highlights
-----------------
1) The constructor fetches the correct checkpoint path through
   `ModelInit.get(<model_id>)`, where `<model_id>` is derived from
   `str(self)` (see `BaseEmbedder.__str__`).  
   *Examples*
       BertBaseEmbedder  ->  "bertbase"
       BertLargeEmbedder ->  "bertlarge"

2) `_get_embeddings()` exposes the `nn.Embedding` module that stores the
   vocabulary lookup weights (`transformer.embeddings.word_embeddings`),
   which is exactly what `BaseEmbedder.get_embeddings()` expects.
"""

from __future__ import annotations

import abc

from transformers.models.bert import BertModel

from embedder.base_embedder import BaseEmbedder
from utils.config_init import ModelInit


class BertEmbedder(BaseEmbedder, abc.ABC):
    """
    Shared implementation for every BERT flavour.

    The concrete subclasses only differ in the *checkpoint string*
    looked up via `ModelInit`, therefore all heavy lifting is done here.
    """

    def __init__(self) -> None:
        super().__init__()

        # Retrieve the checkpoint name/path using the textual identifier
        # produced by BaseEmbedder.__str__()
        model_name: str = ModelInit.get(str(self))

        # Load weights with HuggingFace transformers
        self.transformer: BertModel = BertModel.from_pretrained(model_name)

    # ------------------------------------------------------------------ #
    # BaseEmbedder abstract method                                        #
    # ------------------------------------------------------------------ #
    def _get_embeddings(self):
        """
        Return the `torch.nn.Embedding` layer that stores token embeddings.
        """
        return self.transformer.embeddings.word_embeddings


class BertBaseEmbedder(BertEmbedder):
    """BERT-BASE (12-layer, 110M parameters) implementation."""
    pass


class BertLargeEmbedder(BertEmbedder):
    """BERT-LARGE (24-layer, 340M parameters) implementation."""
    pass
