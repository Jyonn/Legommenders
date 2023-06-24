import json
import os

import numpy as np
import torch
from UniTok import Vocab

image_path = '/data3/qijiong/Data/MIND-image-cropped'

image_list = os.listdir(image_path)
image_list.sort()  # 'N1.jpg', ...

# remove .jpg
for image in image_list:
    assert image[-4:] == '.jpg'
image_list = [image[:-4] for image in image_list]

image_embed_file = 'image_embeds.pt'
image_embeds = torch.load(image_embed_file)

assert len(image_list) == len(image_embeds)

index_dict = {image: i for i, image in enumerate(image_list)}
nid_vocab = Vocab('nid').load('../data/MIND-small-v2/news')
news_align_dict = json.load(open('news-align.json'))

not_found = 0
embeds = []
for i in range(len(nid_vocab)):
    nid = news_align_dict[str(i)]
    if nid in index_dict:
        embeds.append(image_embeds[index_dict[nid]])
    else:
        embeds.append(torch.zeros(512))
        not_found += 1

print(f'{not_found} images not found')

embeds = torch.stack(embeds, dim=0).numpy()
np.save('image_embeds.npy', embeds)
print(embeds.shape)
