from glob import glob
import random

import numpy as np
import h5py
import cv2
from tqdm import tqdm

from utils.conversion import convert_rgb_to_y

def create_patches(file_paths, output_file, len, PATCH_SIZE=64):

    STRIDE = PATCH_SIZE

    lr_patches, hr_patches = list(), list()

    print(f"Building {output_file}...")

    for lr_path, hr_path in tqdm(file_paths, total=len):

        lr = cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(cv2.imread(hr_path), cv2.COLOR_BGR2RGB)

        height, width, _ = hr.shape
        lr = cv2.resize(lr, (width, height), interpolation=cv2.INTER_CUBIC)

        lr = convert_rgb_to_y(lr)
        hr = convert_rgb_to_y(hr)

        for i in range(0, lr.shape[0] - PATCH_SIZE + 1, STRIDE):
            for j in range(0, lr.shape[1] - PATCH_SIZE + 1, STRIDE):
                lr_patches.append(lr[i:i + PATCH_SIZE, j:j + PATCH_SIZE])
                hr_patches.append(hr[i:i + PATCH_SIZE, j:j + PATCH_SIZE])

    lr_patches, hr_patches = np.array(lr_patches), np.array(hr_patches)

    with h5py.File(output_file, "w") as hf:
        hf.create_dataset("lr", lr_patches)
        hf.create_dataset("hr", hr_patches)

if __name__ == "__main__":
    train_hr = sorted(glob("dataset_main/DIV2K_train_HR/**/*.png", recursive=True))
    train_lr = sorted(glob("dataset_main/DIV2K_train_LR*/**/*.png", recursive=True))

    test_hr = sorted(glob("dataset_main/DIV2K_valid_HR/**/*.png", recursive=True))
    test_lr = sorted(glob("dataset_main/DIV2K_valid_LR*/**/*.png", recursive=True))

    pairs = list(zip(train_lr, train_hr))
    random.shuffle(pairs)

    train_pairs = pairs[:600]
    valid_pairs = pairs[600:]

    create_patches(train_pairs, 600, "train.hdf5")
    create_patches(valid_pairs, 200, "valid.hdf5")
    create_patches(zip(test_lr, test_hr), "test.hdf5", 100)


# for input, label in train_pairs:
#     input_des = input.replace("dataset_main", "dataset")
#     label_des = label.replace("dataset_main", "dataset")
#     os.makedirs(os.path.dirname(input_des), exist_ok=True)
#     os.makedirs(os.path.dirname(label_des), exist_ok=True)
#     shutil.copy(input, input_des)
#     shutil.copy(label, label_des)

# for input, label in valid_pairs:
#     input_des = input.replace("dataset_main", "dataset").replace("train", "valid")
#     label_des = label.replace("dataset_main", "dataset").replace("train", "valid")
#     os.makedirs(os.path.dirname(input_des), exist_ok=True)
#     os.makedirs(os.path.dirname(label_des), exist_ok=True)
#     shutil.copy(input, input_des)
#     shutil.copy(label, label_des)

# for input, label in zip(test_input, test_label):
#     input_des = input.replace("dataset_main", "dataset").replace("valid", "test")
#     label_des = label.replace("dataset_main", "dataset").replace("valid", "test")
#     os.makedirs(os.path.dirname(input_des), exist_ok=True)
#     os.makedirs(os.path.dirname(label_des), exist_ok=True)
#     shutil.copy(input, input_des)
#     shutil.copy(label, label_des)
