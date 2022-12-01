import os
from glob import glob
import random
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from utils.conversion import convert_rgb_to_y, convert_rgb_to_ycbcr

class DIV2K_dataset(torch.utils.data.Dataset):

    def __init__(self, dir):
        random.seed(0)
        self.paths = glob(os.path.join(dir, "*.npz"))
        random.shuffle(self.paths)

    def __getitem__(self, index):

        data = np.load(self.paths[index])

        lr_image = torch.from_numpy(data["lr"])
        hr_image = torch.from_numpy(data["hr"])
         
        lr_image = convert_rgb_to_y(lr_image).unsqueeze(0)
        hr_image = convert_rgb_to_y(hr_image).unsqueeze(0)

        lr_image = lr_image / 255.0
        hr_image = hr_image / 255.0
        
        return lr_image, hr_image

    def __len__(self):
        return len(self.paths)

class test_DIV2K_dataset(torch.utils.data.Dataset):

    def __init__(self, paths):
        random.seed(0)
        hr_paths = sorted(glob(os.path.join(paths[0], "*.png"), recursive=True))
        lr_paths = sorted(glob(os.path.join(paths[1], "*/*.png"), recursive=True))
        self.pair_paths = list(zip(lr_paths, hr_paths))
        random.shuffle(self.pair_paths)

    def __getitem__(self, index):
        lr_path, hr_path = self.pair_paths[index]
        file_name = Path(hr_path).parts[-1]
        lr_image  = cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(cv2.imread(hr_path), cv2.COLOR_BGR2RGB)

        height, width, _ = hr_image.shape
        lr_image = cv2.resize(lr_image, (width, height), interpolation=cv2.INTER_CUBIC)

        # height, width, _ = hr_image.shape
        # lr_image = cv2.resize(lr_image, (width//4, height//4))
        # hr_image = cv2.resize(hr_image, (width//4, height//4))

        lr_image = torch.from_numpy(lr_image)
        hr_image = torch.from_numpy(hr_image)

        lr_image = convert_rgb_to_ycbcr(lr_image)
        hr_image = convert_rgb_to_ycbcr(hr_image)
        
        lr_image = lr_image / 255.0
        hr_image = hr_image / 255.0
        
        return file_name, lr_image, hr_image

    def __len__(self):
        return len(self.pair_paths)
