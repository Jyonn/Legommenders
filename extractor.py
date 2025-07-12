"""
extractor.py

Offline **embedding dump** utility
==================================

The script makes it easy to export the *final* user / item embeddings of
a trained recommender model so that they can be consumed by downstream
applications (e.g. nearest-neighbour search, visualisation).

Design
------
We again reuse the heavy lifting from `BaseLego` which:

    • builds the data pipeline,
    • loads the trained model (if `--exp.load.sign` is set),
    • handles device placement, …

The `Extractor` subclass only adds two helper functions
(`extract_user_embedding`, `extract_item_embedding`) and decides which
one to call based on the `--target` CLI flag.

Implementation details
----------------------
1. *User* embeddings  
   We rely on the trainer’s **fast-eval cache**: `self.cacher.user.repr`
   is populated when `.get_train_loader(Symbols.test)` is called.

2. *Item* embeddings  
   We first make sure the item cache is filled
   (`self.cacher.item.cache()`) and then export the tensor.

Both NumPy arrays are written to the experiment directory (`exp.dir`)
under `<target>_embeddings.npy`.

Example
-------
python extractor.py \
    --data  movielens \
    --model config/model/bert_recommender.yaml \
    --target item \
    --exp   config/exp/default.yaml \
    --embed config/embed/bert.yaml
"""

from __future__ import annotations

import os
import numpy as np

from pigmento import pnt

from base_lego import BaseLego
from loader.symbols import Symbols
from utils.config_init import CommandInit


class Extractor(BaseLego):
    """
    Adds only *one* public method (`run`) on top of BaseLego.
    """

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def extract_user_embedding(self) -> None:
        """
        Dump the cached user representations to
        `<exp.dir>/user_embeddings.npy`.
        """
        # Ensure the cache is materialized (same mechanism used during fast
        # evaluation). `Symbols.test` flag prevents gradient tracking, etc.
        self.manager.get_train_loader(Symbols.test)
        assert self.cacher.user.cached, 'Fast eval not enabled – user cache empty.'

        user_embeddings = self.cacher.user.repr.detach().cpu().numpy()
        store_path = os.path.join(self.exp.dir, 'user_embeddings.npy')
        pnt(f'Store user embeddings to {store_path}')
        np.save(store_path, user_embeddings)

    def extract_item_embedding(self) -> None:
        """
        Dump (and if necessary first *compute*) item representations to
        `<exp.dir>/item_embeddings.npy`.
        """
        # Trigger creation of the item cache
        self.cacher.item.cache(self.resampler.item_cache)
        item_embeddings = self.cacher.item.repr.detach().cpu().numpy()
        store_path = os.path.join(self.exp.dir, 'item_embeddings.npy')
        pnt(f'Store item embeddings to {store_path}')
        np.save(store_path, item_embeddings)

    # ------------------------------------------------------------------ #
    # BaseLego hook                                                      #
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """
        Dispatch according to `--target {user,item}`.
        """
        target = self.config.target.lower()
        if target == Symbols.user.name:
            self.extract_user_embedding()
        elif target == Symbols.item.name:
            self.extract_item_embedding()
        else:
            raise ValueError(
                f'Unknown target: {self.config.target}. Expect "user" or "item".'
            )


# ---------------------------------------------------------------------- #
# CLI entry point                                                        #
# ---------------------------------------------------------------------- #
if __name__ == '__main__':
    configuration = CommandInit(
        required_args=['data', 'model', 'target'],
        default_args=dict(
            exp='config/exp/default.yaml',
            embed='config/embed/null.yaml',
            hidden_size=256,
            item_hidden_size='${hidden_size}$',
        ),
    ).parse()

    extractor = Extractor(config=configuration)
    extractor.run()
