import torch
from torch import nn

from model_v2.interaction.base_interaction import BaseInteraction


class DotProduct(BaseInteraction):
    @staticmethod
    def predict(user_embedding: torch.Tensor, candidates: torch.Tensor, labels: torch.Tensor):
        # user_embedding: batch_size, embedding_dim
        # candidates: batch_size, 1, embedding_dim
        # labels: batch_size

        candidates = candidates.squeeze(1)  # batch_size, embedding_dim
        scores = torch.sum(user_embedding * candidates, dim=1)  # batch_size
        loss = nn.functional.binary_cross_entropy_with_logits(scores, labels.float())

        return loss
