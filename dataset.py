import os
from glob import glob
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

class DIV2K_dataset(torch.utils.data.Dataset):

    def __init__(self, input_dir, output_dir):
        random.seed(0)
        input_paths  = sorted(glob(os.path.join(input_dir, "**/*.png"), recursive=True))
        output_paths = sorted(glob(os.path.join(output_dir, "**/*.png"), recursive=True))
        self.pair_paths = list(zip(input_paths, output_paths))
        random.shuffle(self.pair_paths)

    def __getitem__(self, index):
        input_path, output_path = self.pair_paths[index]
        # BGR images not RGB
        input_image  = cv2.imread(input_path)
        output_image = cv2.imread(output_path)

        height, width, _ = input_image.shape
        input_image = cv2.resize(input_image, (width//4, height//4))
        output_image = cv2.resize(output_image, (width//4, height//4))

        input_image = torch.from_numpy(input_image)
        output_image = torch.from_numpy(output_image)

        input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
        output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())

        input_image = input_image.permute(2, 0, 1)
        output_image = output_image.permute(2, 0, 1)

        return input_image, output_image

    def __len__(self):
        return len(self.pair_paths)
