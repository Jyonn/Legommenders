import pandas as pd
from UniTok import UniDep, UniTok
from UniTok.tok import IdTok, NumberTok

fake_dir = '../../data/MIND-small-v2/cold'
user_dir = '../../data/MIND-small-v2/user-fake-v2'

fake_depot = UniDep(fake_dir)
user_depot = UniDep(user_dir)

fake = [0] * user_depot.sample_size

for sample in fake_depot:
    uid = fake_depot.vocabs['nid'][sample['nid']].replace('FAKE', 'U')
    uid = uid.split('@')[0]
    uid = user_depot.vocabs['uid'][uid]
    fake[uid] = 1

df = pd.DataFrame(data={
    'uid': list(user_depot.vocabs['uid']),
    'fake': fake,
})

ut = UniTok()
ut.add_col('uid', tok=IdTok(vocab=user_depot.vocabs['uid']))
ut.add_col('fake', tok=NumberTok(name='fake', vocab_size=2))
ut.read(df).analyse()
depot = ut.tokenize().to_unidep()
user_depot.union(depot).export(user_dir)
