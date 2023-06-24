import os
from PIL import Image
from tqdm import tqdm

from utils.gpu import GPU

os.makedirs('/data3/qijiong/Data/MIND-image-cropped', exist_ok=True)

gpu = GPU.auto_choose(torch_format=True)
image_path = '/data3/qijiong/Data/MIND-image'

# 定义矩形区域
left = 15
top = 15
right = 15 + 200
bottom = 15 + 165

# list all images
image_list = os.listdir(image_path)
image_list.sort()
image_list = [os.path.join(image_path, image) for image in image_list]

# open 64 images and extract image embeddings

# save image embeddings

# 打开图片
for path in tqdm(image_list):
    image = Image.open(path)
    cropped_image = image.crop((left, top, right, bottom))
    cropped_image.save(path.replace('MIND-image', 'MIND-image-cropped'))
