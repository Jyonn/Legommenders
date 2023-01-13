import torch
import sys

sys.path.append('..')
from utils.gpu import GPU

path = '/data1/qijiong/Code/PREC/saving/MINDsmall/L3H12E768-B100/mind-mlm-news/epoch_48.bin'

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

torch.save(state_dict, 'PREC-MINDsmall-L3H12E768-epoch_48.bin')

