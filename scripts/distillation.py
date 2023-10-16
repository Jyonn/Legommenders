import argparse
import copy
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from UniTok import UniDep, Fut

sys.path.append('..')
from utils.cluster.kmeans import kmeans


class DataMeta:
    name: str
    base_path: str
    user_depot: str
    item_depot: str
    train_depot = 'train'
    dev_depot = 'dev'
    taste_attr = None
    history_attr = 'history'


class MINDDataMeta(DataMeta):
    name = 'MIND'
    base_path = '../data/MIND-small-v2'
    user_depot = 'user-grp'
    item_depot = 'news'
    taste_attr = 'cat'


class GoodreadsDataMeta(DataMeta):
    name = 'Goodreads'
    base_path = '../data/Goodreads'
    user_depot = 'user'
    item_depot = 'book-desc'


class MovieLensDataMeta(DataMeta):
    name = 'MovieLens'
    base_path = '../data/MovieLens-100k'
    user_depot = 'user'
    item_depot = 'item'


class Distillation:
    def __init__(self, meta: DataMeta, model: str = 'NAML'):
        self.meta = meta
        self.model = model

        self.embedding_path = os.path.join(
            '../saving',
            f'{self.meta.name}-Distillation-All',
            self.model,
            'free-train_get_user_embedding',
            'user_embeddings.npy'
        )

        self.user_embedding = torch.Tensor(np.load(self.embedding_path))
        self.num_users = len(self.user_embedding)

        self.store_dir = os.path.join(
            self.meta.base_path,
            'distillation'
        )

        self.user_depot_dir = os.path.join(self.meta.base_path, self.meta.user_depot)
        self.item_depot_dir = os.path.join(self.meta.base_path, self.meta.item_depot)
        self.train_depot_dir = os.path.join(self.meta.base_path, self.meta.train_depot)
        self.dev_depot_dir = os.path.join(self.meta.base_path, self.meta.dev_depot)

        self.user_depot = UniDep(self.user_depot_dir)
        self.item_depot = UniDep(self.item_depot_dir)
        self.train_depot = UniDep(self.train_depot_dir)
        self.dev_depot = UniDep(self.dev_depot_dir)

        self.topic_embed_path = os.path.join(
            '/home/qijiong/Code/GENRE-requests/data',
            self.meta.name.lower(),
            'topic_embeddings.npy'
        )
        self.topic_embedding = torch.Tensor(np.load(self.topic_embed_path))

        self.dim = self.user_embedding.shape[1]

    def neg_sampling(self, history):
        taste = set()
        if self.meta.taste_attr:
            for iid in history:
                taste.add(self.item_depot[iid][self.meta.taste_attr])

        user_neg = set()
        while len(user_neg) < 20 + random.randint(1, 5):
            iid = random.randint(0, len(self.item_depot) - 1)
            if self.meta.taste_attr and self.item_depot[iid][self.meta.taste_attr] in taste:
                continue
            user_neg.add(iid)

        return list(user_neg)

    def load_centers(self, k):
        center_path = os.path.join(self.store_dir, f'{k}_centers.{self.model}.npy')
        # detect if centers exists
        if os.path.exists(center_path):
            centers = torch.Tensor(np.load(center_path))
            if centers.shape[1] == self.dim:
                return centers

        _, centers = kmeans(self.user_embedding, k)  # type: torch.Tensor, torch.Tensor
        np.save(center_path, centers.numpy())
        return centers

    def clustering(self, k, select_num, alpha):
        if k < 0:
            k = self.num_users // -k

        key = f'K{k}_select{select_num}_alpha{alpha}'
        print(f'key: {key}')

        centers = self.load_centers(k)
        dists = torch.cdist(self.user_embedding, centers)
        node_centers = torch.argmin(dists, dim=1)
        group_users = []
        group_histories = []
        group_negs = []

        user_history = copy.deepcopy(self.user_depot.data['history'].tolist())

        for i in range(k):
            # noinspection PyTypeChecker
            indices = torch.where(node_centers == i)[0]

            group_user_embeds = self.user_embedding[indices]
            center = centers[i].unsqueeze(0)
            group_dist = torch.cdist(center, group_user_embeds).squeeze(0)

            group_topic_embeds = self.topic_embedding[indices]
            topic_center = group_topic_embeds.mean(dim=0).unsqueeze(0)
            group_topic_dist = torch.cdist(topic_center, group_topic_embeds).squeeze(0)

            group_dist += group_topic_dist * alpha

            sorted_indices = torch.argsort(group_dist)[:select_num]
            sorted_indices = indices[sorted_indices].tolist()
            group_users.extend(sorted_indices)

            history = []
            for uid in sorted_indices:
                history.extend(self.user_depot[uid][self.meta.history_attr])

            for uid in sorted_indices:
                user_history[uid] = history
            group_histories.append(history)
            group_negs.append(self.neg_sampling(history))

        node_centers = node_centers.tolist()

        Fut(
            pd.DataFrame(dict(
                uid=list(range(len(self.user_depot))),
                cid=node_centers,
            )),
            self.user_depot,
            id_col='uid',
        ).construct().store(os.path.join(self.store_dir, f'user_{key}_{self.model}'))

        Fut(
            pd.DataFrame(dict(
                cid=list(range(k)),
                history=group_histories,
                neg=group_negs,
            )),
            self.user_depot,
            id_col='cid',
        ).construct().store(os.path.join(self.store_dir, f'center_{key}_{self.model}'))

        group_users = set(group_users)

        self.user_depot.set_col(
            name='history',
            values=user_history,
        )
        self.user_depot.export(os.path.join(self.store_dir, f'user_keep_{key}_{self.model}'))

        # for depot, mode in zip([train_depot, dev_depot], ['train', 'dev']):
        self.train_depot \
            .filter(bool, col='click') \
            .filter(lambda x: x in group_users, col='uid') \
            .reset_index() \
            .export(os.path.join(self.store_dir, f'train_{key}_{self.model}'))

        self.dev_depot \
            .filter(lambda x: x in group_users, col='uid') \
            .reset_index() \
            .export(os.path.join(self.store_dir, f'dev_{key}_{self.model}'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MIND')
    parser.add_argument('--k', type=int, default=10000)
    parser.add_argument('--select_num', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--model', type=str, default='NAML')
    args = parser.parse_args()

    if args.dataset.lower() == 'MIND'.lower():
        meta = MINDDataMeta()
    elif args.dataset.lower() == 'Goodreads'.lower():
        meta = GoodreadsDataMeta()
    elif args.dataset.lower() == 'MovieLens'.lower():
        meta = MovieLensDataMeta()
    else:
        raise NotImplementedError

    distillation = Distillation(meta, model=args.model)
    distillation.clustering(args.k, args.select_num, args.alpha)
