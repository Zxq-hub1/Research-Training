import os
import torchvision
import numpy as np
import torch.nn
from torch.utils.data import DataLoader
import time
import torch.optim as optim
import argparse
from PIL import Image
from models import UNet
from utils import *
from datetime import datetime
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

file_path = './data/Test/Set14/zebra.png'


model_dir = './runs/Noise2Noisegaussian/checkpoints/model_at_epoch_100.dat'
checkpoint = torch.load(model_dir)

model = UNet()
model = load_model(model, checkpoint).cuda()
model.eval()

image = Image.open(file_path)

# Convert to RGB if the image is not in RGB format
image = image.convert('RGB')

# Get random coordinates for cropping
width, height = image.size

# --- 注释随机裁剪部分 ---
# if width < 256 or height < 256:
#     # Resize the image to ensure it is at least 256x256
#     image = image.resize((256, 256), Image.BICUBIC)
# if height-256>0:
#     top = np.random.randint(0, height - 256)
# else:
#     top=0
# if width-256>0:
#     left = np.random.randint(0, width - 256)
# else:
#     left=0
#
# # Crop the image
# image = image.crop((left, top, left + 256, top + 256))

# 改成固定 256×256 中心 crop（或干脆 resize 成 256×256）
image = image.resize((256, 256), Image.BICUBIC)

# Convert image to tensor
image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)
source,target=get_noisePair(image,noise_type='gaussian',mode=None)

image = image.unsqueeze(0)
source = source.unsqueeze(0)
target = target.unsqueeze(0)

source = source.cuda()
target = target.cuda()
image = image.cuda()

with torch.no_grad():

    pred = model(source)

p1=psnr(source,image)
p2=psnr(target,image)
p3=psnr(pred,image)
show_rgb_image(source,target,pred,
               'Noise1:{:.3f}dB'.format(p1),
               'Noise2:{:.3f}dB'.format(p2),
               'Result:{:.3f}dB'.format(p3))

plt.suptitle('Noise2Noise', fontsize=16)   # 添加大标题
plt.show()


