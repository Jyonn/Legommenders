import abc

import torch

from loader.global_setting import Setting
from model.recommenders.base_recommender import BaseRecommender, BaseRecommenderConfig


class BaseNegRecommenderConfig(BaseRecommenderConfig):
    def __init__(
            self,
            neg_count,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.neg_count = neg_count


class BaseNegRecommender(BaseRecommender, abc.ABC):
    use_neg_sampling = True
    config_class = BaseNegRecommenderConfig
    config: BaseNegRecommenderConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.neg_count = self.config.neg_count

    def predict(self, user_embedding, candidates, batch):
        user_embedding = user_embedding.unsqueeze(1)  # batch_size, 1, embedding_dim
        scores = torch.sum(user_embedding * candidates, dim=2).to(Setting.device)  # batch_size, K+1

        if Setting.status.is_testing:
            return scores

        labels = torch.zeros(scores.shape[0], dtype=torch.long).to(Setting.device)
        return torch.nn.functional.cross_entropy(scores, labels)
