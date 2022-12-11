#!/bin/bash

mkdir dataset_main
cd dataset_main
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip --no-check-certificate
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip --no-check-certificate
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip --no-check-certificate
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip --no-check-certificate

unzip DIV2K_train_LR_bicubic_X2.zip
unzip DIV2K_valid_LR_bicubic_X2.zip
unzip DIV2K_train_HR.zip
unzip DIV2K_valid_HR.zip

cd ..
python3 dataset_prep.py