import abc

import numpy as np
from torch import nn


class BaseEmbedder(abc.ABC):
    transformer: object

    def __init__(self, key):
        self.key = key

    def _get_embeddings(self) -> nn.Embedding:
        raise NotImplementedError

    def get_embeddings(self) -> np.ndarray:
        embedding = self._get_embeddings()
        return embedding.weight.data.cpu().numpy()
