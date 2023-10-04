import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from UniTok import UniDep, Fut


parser = argparse.ArgumentParser()
parser.add_argument('--ItemK', type=int, default=12000)
parser.add_argument('--UserK', type=int, default=None)
parser.add_argument('--select', type=int, default=None)
parser.add_argument('--alpha', type=float, default=None)

args = parser.parse_args()

ItemK = args.ItemK
UserK = args.UserK
select_num = args.select
alpha = args.alpha

if UserK is None or select_num is None or alpha is None:
    assert UserK is None and select_num is None and alpha is None, 'UserK, select_num, alpha must be set together'
    user_key = ''
    key = f'ItemK{ItemK}'
    train_depot_dir = '../data/MIND-small-v2/train'
    dev_depot_dir = '../data/MIND-small-v2/dev'
    user_depot_dir = '../data/MIND-small-v2/user-grp'
    neg_depot_dir = '../data/MIND-small-v2/rd-neg'
    user_depot = UniDep(user_depot_dir)
    neg_depot = UniDep(neg_depot_dir)
    center_depot = None
else:
    user_key = f'K{UserK}_select{select_num}_alpha{alpha}'
    key = f'{user_key}_ItemK{ItemK}'
    train_depot_dir = f'../data/MIND-small-v2/distillation/train_{user_key}'
    dev_depot_dir = f'../data/MIND-small-v2/distillation/dev_{user_key}'
    center_depot_dir = f'../data/MIND-small-v2/distillation/center_{user_key}'
    center_depot = UniDep(center_depot_dir)
    user_depot = neg_depot = None

item_embed_path = '../saving/MIND-small-v2/NAML/free-train_get_item_embedding/item_embeddings.npy'
center_path = item_embed_path.replace('.npy', f'_centers_{ItemK}.npy')
news_depot_dir = '../data/MIND-small-v2/news-v2'
item_depot_export_base_dir = '../data/MIND-small-v2/distillation'
os.makedirs(item_depot_export_base_dir, exist_ok=True)

item_embeds = torch.Tensor(np.load(item_embed_path))
centers = torch.Tensor(np.load(center_path))
news_depot = UniDep(news_depot_dir)
news_num = len(news_depot)
train_depot = UniDep(train_depot_dir)
dev_depot = UniDep(dev_depot_dir)


dist = torch.cdist(item_embeds, centers)
node_to_center = torch.argmin(dist, dim=1)
center_items = []
center_data = dict(
    cid=list(range(ItemK)),
)

for i in range(ItemK):
    indices = torch.where(node_to_center == i)[0]

    group_user_embeds = item_embeds[indices]
    center = centers[i].unsqueeze(0)
    group_dist = torch.cdist(center, group_user_embeds).squeeze(0)

    min_index = torch.argmin(group_dist)
    center_item = news_depot[indices[min_index].item()]
    center_items.append(center_item['nid'])
    for col in center_item:
        if col == news_depot.id_col:
            continue
        if col not in center_data:
            center_data[col] = []
        center_data[col].append(center_item[col])

node_to_center = node_to_center.tolist()

item_center_fut = Fut(
    pd.DataFrame(dict(
        nid=list(range(len(news_depot))),
        cid=node_to_center,
    )),
    news_depot,
    id_col='nid',
).construct().store(os.path.join(item_depot_export_base_dir, f'item_{key}'))

center_fut = Fut(
    pd.DataFrame(center_data),
    news_depot,
    id_col='cid',
).store(os.path.join(item_depot_export_base_dir, f'center_{key}'))

center_items = set(center_items)

# for depot, mode in zip([train_depot, dev_depot], ['train', 'dev']):
train_depot \
    .filter(bool, col='click')\
    .filter(lambda x: x in center_items, col='nid')\
    .reset_index()\
    .export(os.path.join(item_depot_export_base_dir, f'train_{key}'))

dev_depot \
    .filter(lambda x: x in center_items, col='nid')\
    .reset_index()\
    .export(os.path.join(item_depot_export_base_dir, f'dev_{key}'))


item_depot = UniDep(os.path.join(item_depot_export_base_dir, f'item_{key}'))
item_center_depot = UniDep(os.path.join(item_depot_export_base_dir, f'center_{key}'))
item_depot.union(item_center_depot)

histories = []
negs = []

target_depot = user_depot if UserK is None else center_depot
for sample in target_depot:
    history = sample['history']  # type: list
    history = list(filter(lambda x: x in center_items, history))
    histories.append(history)

    taste = set()
    center_taste = set()
    for nid in history:
        taste.add(item_depot[nid]['cat'])
        center_taste.add(node_to_center[nid])
    user_neg = set()
    while len(user_neg) < 20 + random.randint(1, 5):
        nid = random.randint(0, news_num - 1)
        if news_depot[nid]['cat'] not in taste and node_to_center[nid] not in center_taste:
            user_neg.add(nid)
    negs.append(list(user_neg))


if UserK:
    Fut(
        pd.DataFrame(dict(
            cid=list(range(len(center_depot))),
            history=histories,
            neg=negs,
        )),
        center_depot,
        id_col='cid',
    ).store(os.path.join(item_depot_export_base_dir, f'usercenter_{key}'))
else:
    Fut(
        pd.DataFrame(dict(
            uid=list(range(len(user_depot))),
            history=histories,
            neg=negs,
        )),
        user_depot,
        id_col='uid',
    ).store(os.path.join(item_depot_export_base_dir, f'user_{key}'))
    Fut(
        pd.DataFrame(dict(
            uid=list(range(len(user_depot))),
            history=histories,
            neg=negs,
        )),
        user_depot,
        id_col='uid',
    ).store(os.path.join(item_depot_export_base_dir, f'usercenter_{key}'))