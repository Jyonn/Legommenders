import torch

from loader.global_setting import Setting
from model.inputer.base_inputer import BaseInputer
from utils.stacker import Stacker
from utils.torch_pager import TorchPager


class FastDocPager(TorchPager):
    def __init__(
            self,
            inputer: BaseInputer,
            hidden_size,
            llm_skip,
            **kwargs,
    ):
        super().__init__(features=['embeddings', 'mask'], **kwargs)

        self.inputer = inputer
        self.hidden_size = hidden_size
        self.llm_skip = llm_skip
        self.fast_doc_repr = torch.zeros(len(self.contents), hidden_size, dtype=torch.float).to(Setting.device)
        self.stacker = Stacker(aggregator=torch.stack)

    def get_features(self, content, index) -> dict:
        if self.llm_skip:
            return dict(
                embeddings=torch.tensor(index),
                mask=None,
            )
        return dict(
            embeddings=self.inputer.get_embeddings(content),
            mask=self.inputer.get_mask(content),
        )

    def stack_features(self, index):
        features = dict()
        if self.llm_skip:
            features['mask'] = None
        feature_cols = ['embeddings'] if self.llm_skip else self.features

        for feature in feature_cols:
            if isinstance(self.caches[feature][index][0], torch.Tensor):
                features[feature] = torch.stack(self.caches[feature][index]).to(Setting.device)
            else:
                assert isinstance(self.caches[feature][index][0], dict)
                features[feature] = self.stacker(self.caches[feature][index], apply=lambda x: x.to(Setting.device))
        return features

    def combine(self, slices, features, output):
        self.fast_doc_repr[slices] = output.detach()
