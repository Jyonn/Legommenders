import torch
from torch import nn
from torch.nn import functional as F


from loader.global_setting import Setting
from model.operator.miner_operator import PolyAttentionOperator
from model.operator.transformer_operator import TransformerOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender, BaseNegRecommenderConfig


class MINERModelConfig(BaseNegRecommenderConfig):
    def __init__(
            self,
            score_type: str = 'weighted',
            **kwargs,
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


class MINERModel(BaseNegRecommender):
    news_encoder_class = TransformerOperator
    user_encoder_class = PolyAttentionOperator
    config_class = MINERModelConfig
    config: MINERModelConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_aware_attention = TargetAwareAttention(self.config.hidden_size)
        self.use_fast_user_caching = False

    def fuse_user_plugin(self, batch, user_embedding):
        if self.user_plugin:
            shape = user_embedding.shape  # batch_size, num_context_codes, hidden_size
            user_embedding = user_embedding.view(-1, shape[-1])
            uids = batch[self.user_col]
            uids = uids.unsqueeze(dim=1).repeat(1, shape[1]).view(-1)
            user_embedding = self.user_plugin(uids, user_embedding)
            user_embedding = user_embedding.view(shape)
        return user_embedding

    def predict(self, user_embedding, candidates, batch):
        """
        @param user_embedding: batch_size, num_context_codes, hidden_size
        @param candidates: batch_size, K+1, hidden_size
        @return:
        """
        scores = torch.matmul(candidates, user_embedding.permute(0, 2, 1))  # batch_size, K+1, num_context_codes
        if self.config.score_type == 'weighted':
            scores = self.target_aware_attention.forward(
                query=user_embedding,
                key=candidates,
                value=scores
            )
        elif self.config.score_type == 'max':
            scores = scores.max(dim=2)[0]
        elif self.config.score_type == 'mean':
            scores = scores.mean(dim=2)
        else:
            raise ValueError('Unknown score type: {}'.format(self.config.score_type))

        if Setting.status.is_testing or (Setting.status.is_evaluating and not Setting.simple_dev):
            return scores

        labels = torch.zeros(scores.shape[0], dtype=torch.long).to(Setting.device)
        return torch.nn.functional.cross_entropy(scores, labels)
