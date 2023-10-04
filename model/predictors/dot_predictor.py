import torch

from model.predictors.base_predictor import BasePredictor


class DotPredictor(BasePredictor):
    def predict(self, user_embeddings, item_embeddings):
        return torch.sum(user_embeddings * item_embeddings, dim=-1)  # batch_size * (K+1)
