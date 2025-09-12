import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random
from PIL import Image
import torch
import numpy as np
from utils import get_noisePair
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir,noise_type='gaussian',mode=None):
        self.data_dir = data_dir
        self.image_files = os.listdir(data_dir)
        self.noise_type=noise_type
        self.mode=mode

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name)

        # Convert to RGB if the image is not in RGB format
        image = image.convert('RGB')

        # Get random coordinates for cropping
        width, height = image.size
        
        if width < 256 or height < 256:
            # Resize the image to ensure it is at least 256x256
            image = image.resize((256, 256), Image.BICUBIC)
        if height-256>0:
            top = np.random.randint(0, height - 256)
        else:
            top=0
        if width-256>0:
            left = np.random.randint(0, width - 256)
        else:
            left=0

        # Crop the image
        image = image.crop((left, top, left + 256, top + 256))

        # Convert image to tensor
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)  # Convert to CHW format
        source, target = get_noisePair(image, noise_type=self.noise_type, mode=self.mode)

        return_dict = {
            "source": source,
            "target": target,
        }

        return return_dict
