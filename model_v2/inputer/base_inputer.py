from typing import Optional, List, Dict

import torch
from UniTok import Vocab, UniDep

from loader.depot.depot_cache import DepotCache
from model_v2.common.base_config import BaseInputerConfig
from model_v2.utils.embedding_manager import EmbeddingManager


class BaseInputer:
    """
    four cases:
    1. candidate news content (title, category) -> 5 x 64
    2. user clicks (title, category) -> 20 x 64
    3. user clicks (news ids) -> 20 x 64
    4. user clicks (title, category) -> 20 x 64 -> 64
    """
    def __init__(self, config: BaseInputerConfig):
        self.depot = DepotCache.get(config.depot)  # type: UniDep
        self.order = config.order  # type: list

    def get_vocabs(self) -> Optional[List[Vocab]]:
        raise NotImplementedError

    def sample_rebuilder(self, sample: dict):
        raise NotImplementedError

    def get_embeddings(
            self,
            batched_samples: Dict[str, torch.Tensor],
            embedding_manager: EmbeddingManager,
    ):
        raise NotImplementedError

    def embedding_processor(self, embeddings: torch.Tensor, mask: torch.Tensor = None):
        raise NotImplementedError
