import torch

from model.predictors.base_predictor import BasePredictor


class DotPredictor(BasePredictor):
    def predict(self, user_embeddings, item_embeddings):
        # user_embeddings: [B, D]
        # item_embeddings: [B, K+1, D]
        return torch.sum(user_embeddings * item_embeddings, dim=-1)  # [B, K+1]
