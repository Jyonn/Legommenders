from typing import Optional, List, Dict

import torch
from unitok import UniTok
from unitok import Vocab

from loader.embedding_hub import EmbeddingHub


class BaseInputer:
    output_single_sequence = True

    """
    four cases:
    1. candidate item content (title, category) -> 5 x 64
    2. user clicks (title, category) -> 20 x 64
    3. user clicks (item ids) -> 20 x 64
    4. user clicks (title, category) -> 20 x 64 -> 64
    """
    def __init__(self, ut, inputs, embedding_hub: EmbeddingHub, **kwargs):
        self.ut: UniTok = ut
        self.inputs: list = inputs
        self.embedding_hub: EmbeddingHub = embedding_hub

    def get_vocabs(self) -> Optional[List[Vocab]]:
        return []

    def sample_rebuilder(self, sample: dict):
        raise NotImplementedError

    def get_mask(self, batched_samples: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def get_embeddings(
            self,
            batched_samples: Dict[str, torch.Tensor],
    ):
        raise NotImplementedError

    def __call__(self, sample: dict):
        return self.sample_rebuilder(sample)
