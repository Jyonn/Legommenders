"""
base_embedder.py

Abstract helper class that provides a *uniform* interface for exporting
(learned) embedding matrices as NumPy arrays.

Sub-classes are expected to wrap a particular backbone model
(e.g. BERT, GPT-2, word2vec, …).  Each concrete implementation simply
has to override the protected `_get_embeddings()` method so that it
returns the `torch.nn.Embedding` module that holds the parameters to be
exported.

Key points
----------
• Holds a reference to the actual backbone in `self.transformer`
  (type intentionally generic – could be a HuggingFace model, a fairseq
  module, etc.).

• `.get_embeddings()` performs the conversion
      torch.Embedding → float32 NumPy array
  so callers do not have to depend on PyTorch.

• `__str__` produces a short identifier (e.g. "bert" for
  "BertEmbedder"), handy for logging or when saving artefacts to disk.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import torch
from torch import nn


class BaseEmbedder(abc.ABC):
    """
    Abstract base class for *embedding extractors*.

    Concrete subclasses *must* implement `_get_embeddings` and are
    encouraged to populate `self.transformer` with the model instance
    that owns the embedding table.  This enables checkpointing or
    further fine-tuning outside of this helper class.
    """

    # We purposely annotate this as `Any`: different libraries expose
    # different model classes, so we leave it flexible.  Subclasses can
    # refine the type if they wish.
    transformer: Any

    # ------------------------------------------------------------------ #
    # Methods to be implemented by subclasses                             #
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def _get_embeddings(self) -> nn.Embedding:
        """
        Return the `torch.nn.Embedding` object that stores the embeddings.

        Must be provided by every subclass.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Public utility                                                      #
    # ------------------------------------------------------------------ #
    def get_embeddings(self) -> np.ndarray:
        """
        Export the embedding weights as a *CPU float32* NumPy array.

        Returns
        -------
        np.ndarray
            A 2-D array of shape (vocab_size, embedding_dim).
        """
        embedding_layer = self._get_embeddings()                      # torch.nn.Embedding
        # 1. `.weight` accesses the parameter tensor
        # 2. `.data` detaches it from the autograd graph
        # 3. `.to(torch.float32)` ensures a common dtype
        # 4. `.cpu()` moves it to host memory
        # 5. `.numpy()` converts it to a NumPy array
        return embedding_layer.weight.data.to(torch.float32).cpu().numpy()

    # ------------------------------------------------------------------ #
    # Pretty printing                                                     #
    # ------------------------------------------------------------------ #
    def __str__(self) -> str:
        """
        Return a lowercase identifier derived from the class name.

        Example
        -------
        >>> str(BertEmbedder())
        'bert'
        """
        return self.__class__.__name__.replace("Embedder", "").lower()
