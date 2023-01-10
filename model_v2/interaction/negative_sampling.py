import torch

from loader.global_setting import Setting
from model_v2.interaction.base_interaction import BaseInteraction


class NegativeSampling(BaseInteraction):
    @staticmethod
    def predict(user_embedding: torch.Tensor, candidates: torch.Tensor, labels: torch.Tensor = None):
        # user_embedding: batch_size, embedding_dim
        # candidates: batch_size, K+1, embedding_dim, positive sample is the first one
        # labels: unused

        user_embedding = user_embedding.unsqueeze(1)  # batch_size, 1, embedding_dim
        scores = torch.sum(user_embedding * candidates, dim=2).to(Setting.device)  # batch_size, K+1
        labels = torch.zeros(scores.shape[0], dtype=torch.long).to(Setting.device)
        loss = torch.nn.functional.cross_entropy(scores, labels)

        return loss
