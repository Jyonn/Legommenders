# import numpy as np
# from UniTok import UniDep, Vocab
#
# news_dir = '../../data/MIND-small-v2/news-v2'
# user_dir = '../../data/MIND-small-v2/user'
#
# fake_dir = '../../data/MIND-small-v2/cold'
# final_news_dir = '../../data/MIND-small-v2/news-v2-fake'
# final_user_dir = '../../data/MIND-small-v2/user-fake'
#
# fake_depot = UniDep(fake_dir)
#
# news_depot = UniDep(news_dir)
# user_depot = UniDep(user_dir)
#
# # 1. add fake news to news depot
#
# data = news_depot.data
# columns = ['title', 'abs', 'cat']
#
# for col in columns:
#     data[col] = np.array(data[col].tolist() + fake_depot.data[col].tolist())
#
# data['newtitle'] = np.array(data['newtitle'].tolist() + fake_depot.data['title'].tolist())
#
# news_size = news_depot.sample_size
# fake_size = fake_depot.sample_size
# data['nid'] = np.array(data['nid'].tolist() + list(range(news_size, news_size + fake_size)))
#
# cat_vocab = news_depot.vocabs['cat']  # type: Vocab
# cat_vocab.append('others')
# subcat_vocab = news_depot.vocabs['subcat']  # type: Vocab
# subcat_vocab.append('others')
# data['subcat'] = np.array(data['subcat'].tolist() + [len(subcat_vocab) - 1] * fake_size)
#
# nid_vocab = news_depot.vocabs['nid']  # type: Vocab
# fake_nid_vocab = fake_depot.vocabs['nid']  # type: Vocab
# nid_vocab.extend(list(fake_nid_vocab))
#
# data['subcat'] = np.array(data['subcat'].tolist() + [len(subcat_vocab) - 1] * fake_size)
#
# news_depot.meta.vocs['cat'].size += 1
# news_depot.meta.vocs['subcat'].size += 1
# news_depot.meta.vocs['nid'].size += fake_size
# news_depot.sample_size += fake_size
#
# news_depot._indexes = list(range(news_depot.sample_size))
#
# news_depot.export(final_news_dir)



import numpy as np
from UniTok import UniDep, Vocab, Voc

news_dir = '../../data/MIND-small-v2/news'
user_dir = '../../data/MIND-small-v2/user'

fake_dirs = ['../../data/MIND-small-v2/cold-5', '../../data/MIND-small-v2/cold-6']
final_news_dir = '../../data/MIND-small-v2/news-cot-two'
final_user_dir = '../../data/MIND-small-v2/user-cot-two'

fake_depots = [UniDep(fake_dir) for fake_dir in fake_dirs]

news_depot = UniDep(news_dir)
user_depot = UniDep(user_dir)

# 1. add fake news to news depot

data = news_depot.data
columns = ['title', 'abs', 'cat']

for col in columns:
    cols = []
    for fake_depot in fake_depots:
        cols.extend(fake_depot.data[col].tolist())
    data[col] = np.array(data[col].tolist() + cols)

news_size = news_depot.sample_size
fake_size = sum([fake_depot.sample_size for fake_depot in fake_depots])
data['nid'] = np.array(data['nid'].tolist() + list(range(news_size, news_size + fake_size)))

cat_vocab = news_depot.vocabs['cat']  # type: Vocab
cat_vocab.append('others')
subcat_vocab = news_depot.vocabs['subcat']  # type: Vocab
subcat_vocab.append('others')
data['subcat'] = np.array(data['subcat'].tolist() + [len(subcat_vocab) - 1] * fake_size)

nid_vocab = news_depot.vocabs['nid']  # type: Vocab
fake_nid_vocabs = [fake_depot.vocabs['nid'] for fake_depot in fake_depots]  # type: list[Vocab]
for fake_nid_vocab in fake_nid_vocabs:
    nid_vocab.extend(list(fake_nid_vocab))

data['subcat'] = np.array(data['subcat'].tolist() + [len(subcat_vocab) - 1] * fake_size)

news_depot.meta.vocs['cat'].size += 1
news_depot.meta.vocs['subcat'].size += 1
news_depot.meta.vocs['nid'].size += fake_size
news_depot.sample_size += fake_size

news_depot._indexes = list(range(news_depot.sample_size))

news_depot.export(final_news_dir)

# 2. add fake news to user depot

base_id = 0
for fake_depot in fake_depots:
    for sample in fake_depot:
        uid = fake_depot.vocabs['nid'][sample['nid']].replace('FAKE', 'U')  # type: str
        uid = uid.split('@')[0]
        uid = user_depot.vocabs['uid'][uid]
        user_depot.data['history'][uid].append(sample['nid'] + news_size + base_id)
    base_id += fake_depot.sample_size

user_depot.meta.vocs['nid'].size += fake_size
user_depot.vocabs['nid'].load(final_news_dir)

user_depot.export(final_user_dir)

# 3. add fake news to interaction depot

mode_dir = '../../data/MIND-small-v2/{mode}'
final_mode_dir = '../../data/MIND-small-v2/{mode}-cot-two'

for mode in ['train', 'dev', 'test']:
    depot = UniDep(mode_dir.format(mode=mode))
    depot.vocabs['nid'].load(final_news_dir)
    depot.vocs['nid'].size += fake_size
    depot.export(final_mode_dir.format(mode=mode))

# 4. add fake news to neg depot

neg_dir = '../../data/MIND-small-v2/neg'
final_neg_dir = '../../data/MIND-small-v2/neg-cot-two'

depot = UniDep(neg_dir)
depot.vocabs['nid'].load(final_news_dir)
depot.vocs['nid'].size += fake_size
depot.export(final_neg_dir)
