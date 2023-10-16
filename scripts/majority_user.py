import argparse
import os
import sys
from collections import Counter

import numpy as np
from UniTok import UniDep

sys.path.append('..')
from scripts.distillation import DataMeta, MINDDataMeta, GoodreadsDataMeta, MovieLensDataMeta


class MajorityUser:
    def __init__(self, meta: DataMeta):
        self.meta = meta

        self.store_dir = os.path.join(
            self.meta.base_path,
            'majority_user'
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

    def majority_choice(self, k):
        if k < 0:
            k = self.num_users // -k

        uid = self.train_depot.data['uid'].tolist()
        uid_count = Counter(uid)

        inter_count = []
        for i in range(self.num_users):
            inter_count.append(len(self.user_depot[i]['history']))
            inter_count[i] += uid_count.get(i, 0)

        # get the top k users
        selected_users = np.argsort(inter_count)[-k:].tolist()
        key = k

        self.train_depot \
            .filter(bool, col='click') \
            .filter(lambda x: x in selected_users, col='uid') \
            .reset_index() \
            .export(os.path.join(self.store_dir, f'train_{key}_v2'))

        self.dev_depot \
            .filter(lambda x: x in selected_users, col='uid') \
            .reset_index() \
            .export(os.path.join(self.store_dir, f'dev_{key}_v2'))



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

    majority_user = MajorityUser(meta)
    majority_user.majority_choice(args.k)
