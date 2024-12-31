import torch

from loader.env import Env
from model.inputer.base_inputer import BaseInputer
from utils.stacker import Stacker
from loader.pager.base_pager import BasePager


class FastItemPager(BasePager):
    def __init__(
            self,
            inputer: BaseInputer,
            hidden_size,
            placeholder,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.inputer = inputer
        self.hidden_size = hidden_size
        self.fast_item_repr = placeholder.to(Env.device)
        self.stacker = Stacker(aggregator=torch.stack)

    def get_features(self, content, index) -> dict:
        if Env.llm_cache:
            return dict(
                embeddings=torch.tensor(index),
                mask=None,
            )
        return dict(
            embeddings=self.inputer.get_embeddings(content),
            mask=self.inputer.get_mask(content),
        )

    def stack_features(self):
        features = dict()
        if Env.llm_cache:
            features['mask'] = None
        feature_cols = ['embeddings'] if Env.llm_cache else self.current

        for feature in feature_cols:
            if isinstance(self.current[feature][0], torch.Tensor):
                features[feature] = torch.stack(self.current[feature]).to(Env.device)
            else:
                assert isinstance(self.current[feature][0], dict)
                features[feature] = self.stacker(self.current[feature], apply=lambda x: x.to(Env.device))
        return features

    def combine(self, slices, features, output):
        self.fast_item_repr[slices] = output.detach()
