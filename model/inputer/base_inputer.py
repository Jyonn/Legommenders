from typing import Optional, List, Dict

import torch
from UniTok import Vocab, UniDep

from loader.embedding.embedding_hub import EmbeddingHub
from loader.data_hub import DataHub


class BaseInputer:
    output_single_sequence = True

    """
    four cases:
    1. candidate item content (title, category) -> 5 x 64
    2. user clicks (title, category) -> 20 x 64
    3. user clicks (item ids) -> 20 x 64
    4. user clicks (title, category) -> 20 x 64 -> 64
    """
    def __init__(self, hub: DataHub, embedding_manager: EmbeddingHub, **kwargs):
        self.depot = hub.depot  # type: UniDep
        self.order = hub.order  # type: list
        self.embedding_manager = embedding_manager  # type: EmbeddingHub

    def get_vocabs(self) -> Optional[List[Vocab]]:
        raise NotImplementedError

    def sample_rebuilder(self, sample: dict):
        raise NotImplementedError

    def get_mask(self, batched_samples: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def get_embeddings(
            self,
            batched_samples: Dict[str, torch.Tensor],
    ):
        raise NotImplementedError

    # def embedding_processor(self, embeddings: torch.Tensor, mask: torch.Tensor = None):
    #     raise NotImplementedError
