import os

import torch
import sys

sys.path.append('..')
from utils.gpu import GPU

# path = '/data1/qijiong/Code/PREC/saving/MINDsmall/L3H12E768-B100/mind-mlm-news/epoch_48.bin'
# path = '/data1/qijiong/Code/PREC/saving/MINDlarge/mind-mlm-news/epoch_48.bin'
path = '/data1/qijiong/Code/PREC/saving/MINDsmall/D64/epoch_63.bin'

device = GPU.auto_choose(torch_format=True)

d = torch.load(path, map_location=device)
m = d['model']

state_dict = {}

for k in m:
    if k.startswith('bert.encoder.'):
        state_dict[k.replace('bert.encoder.', 'news_encoder.transformer.encoder.')] = m[k]

state_dict = dict(
    model=state_dict
)

os.makedirs('PREC-MINDsmall-L3H4E64', exist_ok=True)
os.makedirs('PREC-MINDlarge-L3H4E64', exist_ok=True)
os.makedirs('PREC-MINDsmall-L3H12E768', exist_ok=True)
os.makedirs('PREC-MINDlarge-L3H12E768', exist_ok=True)

torch.save(state_dict, 'PREC-MINDsmall-L3H4E64/epoch_63.bin')

