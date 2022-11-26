import os
import sys
from pathlib import Path
import shutil
from glob import glob
import random

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

def clear_directory(directory):
	try:
		shutil.rmtree(directory)
	except FileNotFoundError:
		pass
	os.makedirs(directory)

def normalize(frame, min, max):
	return (frame - min) / (max - min)

def convert_to_HU(data, rescale_slope, rescale_intercept):
    return data * rescale_slope - rescale_intercept

if __name__ == "__main__":
    
    TRUNC_RANGE = (0, 500)
    # RESCALE_SLOPE = 1
    # RESCALE_INTERCEPT = -1024

    input_path = "/StudentWorkArea/ManufacturerNII/*/*/*.nii.gz"
    output_dir = "../dataset"
    output_path = os.path.join(output_dir, "all")
    train_path = os.path.join(output_dir, "train")
    val_path = os.path.join(output_dir, "val")
    test_path = os.path.join(output_dir, "test")
    nFiles = 100

    if sys.argv[1] == "generate":

        print("Generating files...")
        # clear_directory(output_path)

        all_files = glob(input_path)
        file_count, idx = 0, 0
        pbar = tqdm(total=nFiles)

        while file_count < nFiles:

            raw_file = all_files[idx]
            hura_file = raw_file.replace("ManufacturerNII", "ManufacturerNIIHuraProcessed")

            idx += 1
            if not os.path.exists(hura_file):
                continue

            parts = Path(raw_file).parts
            file_name = parts[-3] + "_" + parts[-2]
            raw_data = nib.load(raw_file).get_fdata()
            hura_data = nib.load(hura_file).get_fdata()

            if raw_data.shape != hura_data.shape:
                continue
            
            raw_data = np.where(((raw_data < TRUNC_RANGE[0]) | (raw_data > TRUNC_RANGE[1])), 0, raw_data)
            hura_data = np.where(((hura_data < TRUNC_RANGE[0]) | (hura_data > TRUNC_RANGE[1])), 0, hura_data)

            # raw_data = convert_to_HU(raw_data, RESCALE_SLOPE, RESCALE_INTERCEPT)
            # hura_data = convert_to_HU(hura_data, RESCALE_SLOPE, RESCALE_INTERCEPT)
            
            raw_data = normalize(raw_data, TRUNC_RANGE[0], TRUNC_RANGE[1])
            hura_data = normalize(hura_data, TRUNC_RANGE[0], TRUNC_RANGE[1])
            
            for t in range(raw_data.shape[-1]):
                for s in range(raw_data.shape[-2]):
                    file_path = os.path.join(output_path, f"{file_name}_s{s:02d}_t{t:02d}.npz")	
                    np.savez(file_path, input=raw_data[:, :, s, t], output=hura_data[:, :, s, t])

            pbar.update(1)
            file_count += 1

        pbar.close()
    
    elif sys.argv[1] == "split":

        print("Splitting files...")
        
        clear_directory(train_path)
        clear_directory(val_path)
        clear_directory(test_path)

        all_files_path = glob(os.path.join(output_path, "*"))
        print(f"Dataset length: {len(all_files_path)}")
        random.shuffle(all_files_path)

        test_split = 0.1
        val_split = 0.1

        test = all_files_path[:int(test_split*len(all_files_path))]
        all_files_path = all_files_path[int(test_split*len(all_files_path)):]
        val = all_files_path[:int(val_split*len(all_files_path))]
        all_files_path = all_files_path[int(val_split*len(all_files_path)):]

        def split_files(files_path, category):
            for file in files_path:
                new_path = file.replace("all", category)
                shutil.copyfile(file, new_path)
            print(f"{category} split completed")

        split_files(test, "test")
        split_files(val, "val")
        split_files(all_files_path, "train")

    else:
        raise Exception("Invalid Input")
