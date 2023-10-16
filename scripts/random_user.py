import argparse
import os
import sys

import numpy as np
from UniTok import UniDep

sys.path.append('..')
from scripts.distillation import DataMeta, MINDDataMeta, GoodreadsDataMeta, MovieLensDataMeta


class RandomUser:
    def __init__(self, meta: DataMeta):
        self.meta = meta

        self.store_dir = os.path.join(
            self.meta.base_path,
            'random_user'
        )
        os.makedirs(self.store_dir, exist_ok=True)

        self.user_depot_dir = os.path.join(self.meta.base_path, self.meta.user_depot)
        self.item_depot_dir = os.path.join(self.meta.base_path, self.meta.item_depot)
        self.train_depot_dir = os.path.join(self.meta.base_path, self.meta.train_depot)
        self.dev_depot_dir = os.path.join(self.meta.base_path, self.meta.dev_depot)

        self.user_depot = UniDep(self.user_depot_dir)
        self.item_depot = UniDep(self.item_depot_dir)
        self.train_depot = UniDep(self.train_depot_dir)
        self.dev_depot = UniDep(self.dev_depot_dir)

        self.num_users = len(self.user_depot)

    def random_choice(self, k):
        if k < 0:
            k = self.num_users // -k

        allow_users = []
        for i in range(self.num_users):
            if self.user_depot[i]['history']:
                allow_users.append(i)
        selected_users = np.random.choice(allow_users, k, replace=False)
        key = k

        self.train_depot \
            .filter(bool, col='click') \
            .filter(lambda x: x in selected_users, col='uid') \
            .reset_index() \
            .export(os.path.join(self.store_dir, f'train_{key}'))

        self.dev_depot \
            .filter(lambda x: x in selected_users, col='uid') \
            .reset_index() \
            .export(os.path.join(self.store_dir, f'dev_{key}'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MIND')
    parser.add_argument('--k', type=int, default=10000)
    args = parser.parse_args()

    if args.dataset.lower() == 'MIND'.lower():
        meta = MINDDataMeta()
    elif args.dataset.lower() == 'Goodreads'.lower():
        meta = GoodreadsDataMeta()
    elif args.dataset.lower() == 'MovieLens'.lower():
        meta = MovieLensDataMeta()
    else:
        raise NotImplementedError

    random_user = RandomUser(meta)
    random_user.random_choice(args.k)
