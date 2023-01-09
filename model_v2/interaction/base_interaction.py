import abc

import torch


class BaseInteraction(abc.ABC):
    @staticmethod
    def predict(user_embedding: torch.Tensor, candidates: torch.Tensor, labels: torch.Tensor):
        raise NotImplementedError
