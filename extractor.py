import os

import numpy as np
from pigmento import pnt

from base_lego import BaseLego
from loader.symbols import Symbols
from utils.config_init import CommandInit


class Extractor(BaseLego):
    def extract_user_embedding(self):
        self.manager.get_train_loader(Symbols.test)
        assert self.cacher.user.cached, 'fast eval not enabled'
        user_embeddings = self.cacher.user.repr.detach().cpu().numpy()
        store_path = os.path.join(self.exp.dir, 'user_embeddings.npy')
        pnt(f'store user embeddings to {store_path}')
        np.save(store_path, user_embeddings)

    def extract_item_embedding(self):
        self.cacher.item.cache(self.resampler.item_cache)
        item_embeddings = self.cacher.item.repr.detach().cpu().numpy()
        store_path = os.path.join(self.exp.dir, 'item_embeddings.npy')
        pnt(f'store item embeddings to {store_path}')
        np.save(store_path, item_embeddings)

    def run(self):
        if self.config.target.lower() is Symbols.user.name:
            return self.extract_user_embedding()
        elif self.config.target.lower() is Symbols.item.name:
            return self.extract_item_embedding()
        raise ValueError(f'unknown target: {self.config.target}, expect "user" or "item"')


if __name__ == '__main__':
    configuration = CommandInit(
        required_args=['data', 'model', 'exp', 'target'],
        default_args=dict(
            embed='config/embed/null.yaml',
            hidden_size=256,
            item_hidden_size='${hidden_size}$',
        ),
    ).parse()

    extractor = Extractor(config=configuration)
    extractor.run()
