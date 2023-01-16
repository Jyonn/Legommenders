import torch
import sys

from UniTok import UniDep

sys.path.append('..')
from utils.gpu import GPU

# path = '/data1/qijiong/Code/PREC/saving/MINDsmall/L3H12E768-B100/mind-mlm-news/epoch_48.bin'
path = '/data1/qijiong/Code/PREC/saving/MINDlarge/mind-mlm-news/epoch_48.bin'

device = GPU.auto_choose(torch_format=True)

d = torch.load(path, map_location=device)
m = d['model']


prec_eng_embeds = m['extra_modules.mlm.english.decoder.weight'].cpu().numpy()  # 30522 x 768
prec_cat_embeds = m['extra_modules.mlm.cat.decoder.weight'].cpu().numpy()  # 280 x 768

# prec_depot = UniDep(store_dir='/data1/qijiong/Code/PREC/data/MIND/MINDsmall-rec/news')
# our_depot = UniDep(store_dir='../data/MIND-small-v2/news')

prec_depot = UniDep(store_dir='../data/MIND-large/news')
our_depot = UniDep(store_dir='../data/MIND-large/news')

prec_cat = prec_depot.vocab_depot.get_vocab('cat')
our_cat = our_depot.vocab_depot.get_vocab('cat')

cat_embeds = []

for i in range(our_cat.get_size()):
    cat = our_cat.index2obj[i]
    prec_i = prec_cat.obj2index[cat]
    cat_embeds.append(prec_eng_embeds[prec_i])

import numpy as np

prec_cat_embeds = np.array(cat_embeds)

np.save('PREC-MINDlarge-L3H12E768/cat_embeds.npy', prec_cat_embeds)
np.save('PREC-MINDlarge-L3H12E768/eng_embeds.npy', prec_eng_embeds)
