import abc

import numpy as np
import torch
from torch import nn


class BaseEmbedder(abc.ABC):
    transformer: object

    def _get_embeddings(self) -> nn.Embedding:
        raise NotImplementedError

    def get_embeddings(self) -> np.ndarray:
        embedding = self._get_embeddings()
        return embedding.weight.data.to(torch.float32).cpu().numpy()

    def __str__(self):
        return self.__class__.__name__.replace('Embedder', '').lower()
