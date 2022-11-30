import os
from glob import glob
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

class DIV2K_dataset(torch.utils.data.Dataset):

    def __init__(self, lr_dir, hr_dir, is_test=False):
        random.seed(0)
        lr_paths  = sorted(glob(os.path.join(lr_dir, "**/*.png"), recursive=True))
        hr_paths = sorted(glob(os.path.join(hr_dir, "**/*.png"), recursive=True))
        self.pair_paths = list(zip(lr_paths, hr_paths))
        random.shuffle(self.pair_paths)
        self.is_test = is_test

    def __getitem__(self, index):
        lr_path, hr_path = self.pair_paths[index]
        lr_image  = cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(cv2.imread(hr_path), cv2.COLOR_BGR2RGB)

        height, width, _ = hr_image.shape
        lr_image = cv2.resize(lr_image, (width, height), interpolation=cv2.INTER_CUBIC)

        height, width, _ = hr_image.shape
        lr_image = cv2.resize(lr_image, (width//8, height//8))
        hr_image = cv2.resize(hr_image, (width//8, height//8))

        lr_image = torch.from_numpy(lr_image)
        hr_image = torch.from_numpy(hr_image)


        if self.is_test:
            lr_image = convert_rgb_to_ycbcr(lr_image)
            hr_image = convert_rgb_to_ycbcr(hr_image)
        else:    
            lr_image = convert_rgb_to_y(lr_image)
            hr_image = convert_rgb_to_y(hr_image)
        
        # input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
        # output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
        lr_image = lr_image / 255.0
        hr_image = hr_image / 255.0
        
        return lr_image, hr_image

    def __len__(self):
        return len(self.pair_paths)
