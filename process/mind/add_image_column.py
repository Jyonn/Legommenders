"""
This file requires UniTok>=3.1.2, 2.x is not supported.
"""

import pandas as pd
from UniTok import UniDep, Vocab, UniTok
from UniTok.tok import NumberTok

store_dir = '../../data/MIND-small-v2/news-v2'
temp_dir = '../../data/MIND-small-v2/temp-image'
final_dir = '../../data/MIND-small-v2/news-v2-image'


news_depot = UniDep(store_dir)
nid_vocab = news_depot.vocabs['nid']  # type: Vocab

df = pd.DataFrame(data=dict(
    nid=list(nid_vocab),
    cover=list(range(len(nid_vocab))),
))

ut = UniTok().add_index_col('nid').add_col('cover', tok=NumberTok).read(df).tokenize().store(temp_dir)

temp_depot = UniDep(temp_dir)
news_depot.union(temp_depot).export(final_dir)
