import hashlib
import json

from UniTok import UniDep, Vocab
from tqdm import tqdm


def get_md5(news):
    l = [*news['title'], *news['abs']]
    m = hashlib.md5()
    m.update(str(l).encode('utf-8'))
    return m.hexdigest()


small_depot = UniDep('../data/MIND-small-v2/news')
large_depot = UniDep('data/MIND-large/news')
small_nid = small_depot.vocabs['nid']  # type: Vocab
large_nid = large_depot.vocabs['nid']  # type: Vocab

small_md5 = dict()
large_md5 = dict()

for news in tqdm(small_depot):
    small_md5[news['nid']] = get_md5(news)

for news in tqdm(large_depot):
    large_md5[get_md5(news)] = news['nid']

align_dict = dict()

for nid, md5 in tqdm(small_md5.items()):
    if md5 in large_md5:
        align_dict[nid] = large_nid.i2o[large_md5[md5]]


json.dump(align_dict, open('news-align.json', 'w'), indent=2)
