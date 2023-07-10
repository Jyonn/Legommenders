import torch

from loader.global_setting import Setting
from utils.stacker import Stacker
from utils.torch_pager import TorchPager


class FastUserPager(TorchPager):
    def __init__(
            self,
            hidden_size,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.fast_user_repr = torch.zeros(len(self.contents), hidden_size, dtype=torch.float).to(Setting.device)
        self.stacker = Stacker(aggregator=torch.stack)

    def stack_features(self, index):
        stacked = dict()
        for feature in self.features:
            target = self.caches[feature][index]
            if isinstance(target[0], torch.Tensor):
                stacked[feature] = torch.stack(target).to(Setting.device)
            else:
                stacked[feature] = torch.tensor(target).to(Setting.device)
        return dict(batch=stacked)

    def get_features(self, content, index) -> dict:
        return content

    def combine(self, slices, features, output):
        self.fast_user_repr[slices] = output.detach()
