import torch

from model.predictors.base_predictor import BasePredictor


class DotPredictor(BasePredictor):
    """
    Legommenders:
    {
      "summary": "Scores user-item compatibility with an inner product.",
      "contract": {
        "inputs": [
          {"name": "user", "kind": "vector", "shape": {"type": "tensor", "axes": ["B", "D"]}},
          {"name": "item", "kind": "vector", "shape": {"oneOf": [
            {"type": "tensor", "axes": ["B", "K", "D"]},
            {"type": "tensor", "axes": ["B", "D"]}
          ]}}
        ],
        "outputs": [
          {"name": "score", "shape": {"oneOf": [
            {"type": "tensor", "axes": ["B", "K"]},
            {"type": "tensor", "axes": ["B"]}
          ]}}
        ],
        "requires": [{"left": "user.outputDim", "op": "eq", "right": "item.outputDim"}]
      }
    }
    """

    def predict(self, user_embeddings, item_embeddings):
        # user_embeddings: [B, D]
        # item_embeddings: [B, K+1, D]
        return torch.sum(user_embeddings * item_embeddings, dim=-1)  # [B, K+1]
