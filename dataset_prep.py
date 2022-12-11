import os
from glob import glob
import random

import numpy as np
import h5py
import cv2
from tqdm import tqdm

def create_patches(file_paths, output_dir, len, PATCH_SIZE=64):

    STRIDE = PATCH_SIZE

    print(f"Building {output_dir}...")
    output_dir = os.path.join("dataset", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for idx, (lr_path, hr_path) in tqdm(enumerate(file_paths), total=len):

        lr = cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(cv2.imread(hr_path), cv2.COLOR_BGR2RGB)

        height, width, _ = hr.shape
        lr = cv2.resize(lr, (width, height), interpolation=cv2.INTER_CUBIC)

        for i in range(0, lr.shape[0] - PATCH_SIZE + 1, STRIDE):
            for j in range(0, lr.shape[1] - PATCH_SIZE + 1, STRIDE):
                np.savez_compressed(os.path.join(output_dir, f"{idx}_{i}_{j}"), 
                    lr=cv2.resize(lr[i:i + PATCH_SIZE, j:j + PATCH_SIZE], (PATCH_SIZE//2, PATCH_SIZE//2)),
                    hr=hr[i:i + PATCH_SIZE, j:j + PATCH_SIZE])

if __name__ == "__main__":
    train_hr = sorted(glob("dataset_main/DIV2K_train_HR/*.png", recursive=True))
    train_lr = sorted(glob("dataset_main/DIV2K_train_LR*/**/*.png", recursive=True))

    test_hr = sorted(glob("dataset_main/DIV2K_valid_HR/*.png", recursive=True))
    test_lr = sorted(glob("dataset_main/DIV2K_valid_LR*/**/*.png", recursive=True))
    
    pairs = list(zip(train_lr, train_hr))
    random.shuffle(pairs)

    train_pairs = pairs[:600]
    valid_pairs = pairs[600:]

    create_patches(train_pairs, "train", 600)
    create_patches(valid_pairs, "valid", 200)
    # create_patches(zip(test_lr, test_hr), "test", 100)
