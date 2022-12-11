import os
from glob import glob
import random
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from utils.conversion import RBB2Y, RGB2YUV

class DIV2K_dataset(torch.utils.data.Dataset):

    def __init__(self, dir, upscale=False):
        random.seed(0)
        self.paths = glob(os.path.join(dir, "*.npz"))
        random.shuffle(self.paths)
        self.upscale = upscale

    def __getitem__(self, index):

        data = np.load(self.paths[index])

        lr, hr = data["lr"], data["hr"]

        if self.upscale:
            height, width, _ = hr.shape
            lr = cv2.resize(lr, (width, height), interpolation=cv2.INTER_CUBIC)

        lr_image = torch.from_numpy(lr)
        hr_image = torch.from_numpy(hr)
         
        lr_image = RBB2Y(lr_image).unsqueeze(0)
        hr_image = RBB2Y(hr_image).unsqueeze(0)

        lr_image = lr_image / 255.0
        hr_image = hr_image / 255.0
        
        return lr_image, hr_image

    def __len__(self):
        return len(self.paths)

class test_DIV2K_dataset(torch.utils.data.Dataset):

    def __init__(self, paths, upscale=False):
        random.seed(0)
        hr_paths = sorted(glob(os.path.join(paths[0], "*.png"), recursive=True))
        lr_paths = sorted(glob(os.path.join(paths[1], "*/*.png"), recursive=True))
        self.pair_paths = list(zip(lr_paths, hr_paths))
        random.shuffle(self.pair_paths)
        self.upscale = upscale

    def __getitem__(self, index):
        lr_path, hr_path = self.pair_paths[index]
        file_name = Path(hr_path).parts[-1]
        lr  = cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(cv2.imread(hr_path), cv2.COLOR_BGR2RGB)

        if self.upscale:
            height, width, _ = hr.shape
            lr = cv2.resize(lr, (width, height), interpolation=cv2.INTER_CUBIC)

        lr = torch.from_numpy(lr)
        hr = torch.from_numpy(hr)

        lr = RGB2YUV(lr)
        hr = RGB2YUV(hr)
        
        lr = lr / 255.0
        hr = hr / 255.0
        
        return file_name, lr, hr

    def __len__(self):
        return len(self.pair_paths)
