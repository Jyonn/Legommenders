import torch

from loader.meta import Meta
from utils.stacker import FastStacker
from loader.pager.base_pager import BasePager


class FastUserPager(BasePager):
    def __init__(
            self,
            hidden_size,
            placeholder,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        # self.fast_user_repr = torch.zeros(len(self.contents), hidden_size, dtype=torch.float).to(Meta.device)
        self.fast_user_repr = placeholder.to(Meta.device)
        self.stacker = FastStacker(aggregator=torch.stack)

    def stack_features(self):
        stacked = dict()
        for feature in self.current:
            target = self.current[feature]
            if isinstance(target[0], torch.Tensor):
                stacked[feature] = torch.stack(target).to(Meta.device)
            elif isinstance(target[0], dict):
                stacked[feature] = self.stacker(target)
            else:
                stacked[feature] = torch.tensor(target).to(Meta.device)
        # stacked = self.stacker(self.current)
        return dict(batch=stacked)

    def get_features(self, content, index) -> dict:
        return content

    def combine(self, slices, features, output):
        self.fast_user_repr[slices] = output.detach()
