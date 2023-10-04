import random

from UniTok import UniDep
from tqdm import tqdm


news_depot = UniDep('../data/MIND-small-v2/news')
user_depot = UniDep('../data/MIND-small-v2/user')
neg_depot = UniDep('../data/MIND-small-v2/neg')

news_num = len(news_depot)

user_negs = []

for sample in tqdm(user_depot):
    user_taste = set()
    for nid in sample['history']:
        user_taste.add(news_depot[nid]['cat'])

    user_neg = set()
    while len(user_neg) < 20 + random.randint(1, 5):
        nid = random.randint(0, news_num - 1)
        if news_depot[nid]['cat'] not in user_taste:
            user_neg.add(nid)

    user_negs.append(list(user_neg))


neg_depot.set_col(
    name='neg',
    values=user_negs,
)
neg_depot.export('../data/MIND-small-v2/rd-neg')
