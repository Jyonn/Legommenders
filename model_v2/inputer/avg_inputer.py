from typing import Optional, List, Dict

import torch
from UniTok import Vocab

from loader.global_setting import Setting
from model_v2.inputer.base_inputer import BaseInputer
from model_v2.utils.embedding_manager import EmbeddingManager


class AvgInputer(BaseInputer):
    def get_vocabs(self) -> Optional[List[Vocab]]:
        return []

    @classmethod
    def pad(cls, l: list, max_len: int):
        return l + [Setting.UNSET] * (max_len - len(l))

    def sample_rebuilder(self, sample: dict):
        for col in sample:
            max_len = self.depot.get_max_length(col)
            if max_len:
                sample[col] = self.pad(sample[col], max_len)
            sample[col] = torch.tensor(sample[col])
        return sample

    def get_embeddings(
            self,
            batched_samples: Dict[str, torch.Tensor],
            embedding_manager: EmbeddingManager,
    ):
        input_embeddings = []

        for col in batched_samples:
            # title: batch_size, content_len; category: batch_size
            col_input = batched_samples[col]

            seq = col_input.to(Setting.device)  # type: torch.Tensor
            mask = (seq > Setting.UNSET).long()  # type: torch.Tensor
            seq *= mask

            embedding = embedding_manager(col)(seq)  # batch_size, (content_len,) embedding_dim
            mask = mask.unsqueeze(-1)  # batch_size, (content_len,) 1

            if embedding.dim() == 2:
                embedding = embedding.unsqueeze(1)  # batch_size, 1, embedding_dim
                mask = mask.unsqueeze(1)  # batch_size, 1, 1

            embedding *= mask  # batch_size, content_len, embedding_dim
            embedding = embedding.sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # batch_size, embedding_dim
            input_embeddings.append(embedding)

        input_embeddings = torch.concat(input_embeddings, dim=1)  # batch_size, embedding_dim * len(self.order)
        return input_embeddings

    def embedding_processor(self, embeddings: torch.Tensor, mask: torch.Tensor = None):
        # embeddings: batch_size, seq_len, embedding_dim
        # mask: batch_size, seq_len
        # average pooling with mask
        mask = mask.unsqueeze(-1)  # batch_size, seq_len, 1
        embeddings *= mask  # batch_size, seq_len, embedding_dim
        embeddings = embeddings.sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # batch_size, embedding_dim
        return embeddings
