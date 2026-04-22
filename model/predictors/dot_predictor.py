import torch

from model.predictors.base_predictor import BasePredictor


class DotPredictor(BasePredictor):
    metadata = {
        "id": "Dot",
        "displayName": "Dot Product Predictor",
        "explain": "Scores user-item compatibility with an inner product.",
        "contract": {
            "inputShape": "user: tensor[B, D], item: tensor[B, K, D] or tensor[B, D]",
            "outputShape": "tensor[B, K] or tensor[B]",
            "objective": "matching or ranking",
            "requires": ["user.outputDim == item.outputDim"],
        },
    }

    def predict(self, user_embeddings, item_embeddings):
        # user_embeddings: [B, D]
        # item_embeddings: [B, K+1, D]
        return torch.sum(user_embeddings * item_embeddings, dim=-1)  # [B, K+1]
