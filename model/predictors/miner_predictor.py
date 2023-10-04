import torch
from torch import nn
from torch.nn import functional as F

from model.predictors.base_predictor import BasePredictorConfig, BasePredictor


class MINERPredictorConfig(BasePredictorConfig):
    def __init__(
            self,
            score_type: str = 'weighted',
            **kwargs
    ):
        super().__init__(**kwargs)

        self.score_type = score_type


class TargetAwareAttention(nn.Module):
    """Implementation of target-aware attention network"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
        @param query: batch_size, num_context_codes, hidden_size
        @param key: batch_size, K+1, hidden_size
        @param value: batch_size, K+1, num_context_codes
        @return: batch_size, K+1
        """
        proj = F.gelu(self.linear(query))  # batch_size, num_context_codes, hidden_size
        weights = F.softmax(torch.matmul(key, proj.permute(0, 2, 1)), dim=2)  # batch_size, K+1, num_context_codes
        outputs = torch.mul(weights, value).sum(dim=2)  # batch_size, K+1
        return outputs


class MINERPredictor(BasePredictor):
    config_class = MINERPredictorConfig
    config: MINERPredictorConfig
    allow_ranking = False
    keep_input_dim = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.target_aware_attention = TargetAwareAttention(self.config.hidden_size)

    def predict(self, user_embeddings, item_embeddings):
        # item_embeddings: batch_size, K+1, hidden_size
        # user_embeddings: batch_size, num_context_codes, hidden_size
        scores = torch.matmul(item_embeddings, user_embeddings.permute(0, 2, 1))  # batch_size, K+1, num_context_codes
        if self.config.score_type == 'weighted':
            return self.target_aware_attention.forward(
                query=user_embeddings,
                key=item_embeddings,
                value=scores
            )
        if self.config.score_type == 'max':
            return scores.max(dim=2)[0]
        if self.config.score_type == 'mean':
            return scores.mean(dim=2)

        raise ValueError(f'Unknown score type: {self.config.score_type}')
