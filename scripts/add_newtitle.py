import numpy as np
from UniTok import UniDep, Col

target_depot = 'data/MIND-small-v2/news-cot-two'
newtitle_dir = 'data/MIND-small-v2/newtitle'

depot = UniDep(target_depot)
newtitle = UniDep(newtitle_dir)


newtitle_col = newtitle.data['newtitle'].tolist()

for i in range(len(newtitle), len(depot)):
    newtitle_col.append(depot.data['title'][i])

newtitle_col = np.array(newtitle_col)

depot.data['newtitle'] = newtitle_col
nt_col = Col('newtitle', voc=depot.vocs['english'], max_length=25)
depot.vocs['english'].cols.append(nt_col)
depot.cols['newtitle'] = nt_col
depot.export('data/MIND-small-v2/news-ABtoC')