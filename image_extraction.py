import os
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from utils.gpu import GPU

gpu = GPU.auto_choose(torch_format=True)
image_path = '/data3/qijiong/Data/MIND-image-cropped'

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# use gpu
model = model.to(gpu)

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# list all images
image_list = os.listdir(image_path)
image_list.sort()
image_list = [os.path.join(image_path, image) for image in image_list]

# open 64 images and extract image embeddings

batch_size = 64
embeds = []

for i in tqdm(range(0, len(image_list), batch_size)):
    start, end = i, min(i + batch_size, len(image_list))
    images = []
    for image in image_list[start:end]:
        images.append(Image.open(image))
    inputs = processor(text=["a photo of a cat"], images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(gpu) for k, v in inputs.items()}
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds.cpu().detach()  # [32, 512]
    embeds.append(image_embeds)


# save image embeddings

import torch
embeds = torch.cat(embeds, dim=0)  # [N, 512]
torch.save(embeds, 'image_embeds.pt')
