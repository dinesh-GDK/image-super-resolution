# Image Super Resolution
Repo to train and test Image Super Resolution models
## Setup

Use conda to create a python virtual environment and install dependencies

```bash
conda create -n <environment_name> python=3.7

# use GPU
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

pip3 install -r requirements.txt
```

Then run the bash script to download and set up data for training

```bash
bash dataset_setup.sh
```

## Train
To train U-Net model run the command
```bash
python3 main.py --mode=train --model=unet
```
To train RRDB-Net model run the command
```bash
python3 main.py --mode=train --model=rrdbnet
```
## Test
To test the trained U-Net model
```bash
python3 main.py --mode=train --model=unet --model_path=results/<YYYY-MM-DD-HH-MM-SS>/models/<model_name>.pt
```
To test the trained RRDB-Net model
```bash
python3 main.py --mode=train --model=rrdbnet --model_path=results/<YYYY-MM-DD-HH-MM-SS>/models/<model_name>.pt
```
## Overview
```
/image-super-resolution
    |
    -- models (contains all the models used to train and test)
    |   |
    |   -- UNet.py (U-Net model impelmentation)
    |   |
    |   -- RRBDNet.py (RRDB-Net model impelmentation)
    |
    -- utils (contains common functions)
    |   |
    |   -- conversion.py (scripts used for color conversion)
    |   |
    |   -- logger.py (logger)
    |
    -- dataset_setup.sh (bash script to download and create dataset (patches))
    |
    -- dataset_prep.py (reads the dataset folder and created dataset (patches))
    |
    -- dataset.py (PyTorch dataset class used for training, validation and testing)
    |
    -- main.py (main file to execute)
    |
    -- requirements.txt (python library dependencies to run the codebase)
```