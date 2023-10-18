import os

import numpy as np
from pigmento import pnt

from loader.meta import Phases
from loader.mode.base_mode import BaseMode


class GetEmbedMode(BaseMode):
    load_model = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cacher = self.legommender.cacher
        self.resampler = self.controller.resampler
        self.store_dir = self.controller.exp.dir

    def get_user_embedding(self):
        self.controller.get_loader(Phases.train).test()
        assert self.cacher.user.cached, 'fast eval not enabled'
        user_embeddings = self.cacher.user.repr.detach().cpu().numpy()
        store_path = os.path.join(self.store_dir, 'user_embeddings.npy')
        pnt(f'store user embeddings to {store_path}')
        np.save(store_path, user_embeddings)

    def get_item_embedding(self):
        self.cacher.item.cache(self.resampler.item_cache)
        item_embeddings = self.cacher.item.repr.detach().cpu().numpy()
        store_path = os.path.join(self.store_dir, 'item_embeddings.npy')
        pnt(f'store item embeddings to {store_path}')
        np.save(store_path, item_embeddings)

    def work(self, *args, target, **kwargs):
        if target == 'user':
            return self.get_user_embedding()
        elif target == 'item':
            return self.get_item_embedding()
        raise ValueError(f'unknown target {target}')
