import os
import warnings
import argparse
import json
from datetime import datetime
import csv
import random
import gc

from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize

from dataset import DIV2K_dataset, test_DIV2K_dataset
from utils.logger import get_logger
from utils.conversion import YUV2RGB
from models import UNet, RRDBNet

class Orchestrator():
    
    def __init__(self, args, config_path=None):
        self.mode = args.mode
        self.num_workers = args.num_workers
        self.update_rate = args.update_rate
        self.save_rate = args.save_rate
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.model_path = args.model_path
        self.weight_decay = args.weight_decay
        self.do_validation = args.do_validation
        self.DEBUG = args.DEBUG

        train_dataset_path = os.path.join(args.data_dir, 'train')
        valid_dataset_path = os.path.join(args.data_dir, 'valid')
        test_dataset_path = args.test_data
    
        if config_path is not None:
            self.get_config(config_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.DEBUG:
            self.result_path = os.path.join(args.result_dir, "sample")
        else:
            self.result_path = os.path.join(args.result_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        self.models_path = os.path.join(self.result_path, "models")

        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(self.result_path, "tensorboard"))
        # log file
        self.logger = get_logger(os.path.join(self.result_path, "log"))

        if args.model == "unet":
            self.model = UNet().to(self.device)
            upscale = True
        elif args.model == "rrdbnet":
            self.model =  RRDBNet(1, 1, 64, 23, gc=32).to(self.device)
            upscale = False
        else:
            raise Exception("Model not available")
        
        self.criterion = torch.nn.MSELoss()

        if self.mode == "train":

            train_data = DIV2K_dataset(train_dataset_path, upscale)
            val_data   = DIV2K_dataset(valid_dataset_path, upscale)
            self.train_dataloader = DataLoader(train_data, self.batch_size,
                                        shuffle=True, num_workers=self.num_workers)
            self.val_dataloader   = DataLoader(val_data, self.batch_size, num_workers=self.num_workers)

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.model = torch.nn.DataParallel(self.model)
            self.train()

        elif self.mode == "test":

            test_data  = test_DIV2K_dataset(test_dataset_path, upscale)
            self.test_dataloader = DataLoader(test_data, 1, num_workers=self.num_workers)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.test()

        else:
            self.logger.error("Invalid Mode")

    def train(self):

        self.logger.info("Start Training...")

        best_epoch_loss = float("inf")
        best_val_loss = float("inf")
        
        for epoch in range(self.epochs):

            torch.cuda.empty_cache()
            self.model.train()

            b="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
            pbar = tqdm(total=len(self.train_dataloader), bar_format=b,
                        desc=f"Epochs: {epoch+1}/{self.epochs}", ascii=" =")
            
            epoch_loss = 0
            for i, (data, label) in enumerate(self.train_dataloader):

                self.optimizer.zero_grad()
                input, target = data.to(self.device), label.to(self.device)
                predict = self.model(input)
                loss = self.criterion(predict, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                if i % self.update_rate == 0:
                    pbar.update(min(self.update_rate, len(self.train_dataloader) - i))
                    pbar.set_postfix(Loss=f"{loss.item():.8f}")
            
            epoch_loss /= len(self.train_dataloader)
            self.writer.add_scalar("Loss/epochs", epoch_loss, epoch)
            pbar.set_postfix(Loss=f"{epoch_loss:.8f}")
            pbar.close()

            if (epoch+1) % self.save_rate == 0:
                self.save_model(f"epoch{epoch+1:04d}")

            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                self.save_model("best_train_model")

            if self.do_validation:
                val_loss = self.validate(epoch)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model("best_val_model")

            self.logger.info(f"Epoch {epoch+1} complete")
            
        self.logger.info("Training complete")

    def validate(self, epoch):
        
        torch.cuda.empty_cache()
        self.model.eval()

        b="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
        pbar = tqdm(total=len(self.val_dataloader), bar_format=b,
                    desc=f"Validation", ascii=" -")
        
        val_loss = 0
        with torch.no_grad():
            for i, (data, label) in enumerate(self.val_dataloader):

                input, target = data.to(self.device), label.to(self.device)
                predict = self.model(input)
                loss = self.criterion(predict, target)

                val_loss += loss.item()
                
                if i % self.update_rate == 0:
                    pbar.update(min(self.update_rate, len(self.val_dataloader) - i))
                    pbar.set_postfix(Loss=f"{loss.item():.8f}")
        
        val_loss /= len(self.val_dataloader)
        self.writer.add_scalar("Val Loss/epochs", val_loss, epoch)
        pbar.set_postfix(Loss=f"{val_loss:.8f}")
        pbar.close()

        return val_loss

    def test(self):

        self.logger.info("Start Testing...")

        torch.cuda.empty_cache()
        self.model.eval()

        test_idx = random.sample(range(0, len(self.test_dataloader)), 10)

        b="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
        pbar = tqdm(total=len(self.test_dataloader), bar_format=b,
                    desc=f"Test", ascii=" >")

        f = open(os.path.join(self.result_path, "_metrics.csv"), "w+")
        csv_writer = csv.writer(f)
        csv_writer.writerow(["S.No", "Original RMSE", "Predict RMSE", \
            "Original PSNR", "Predict PSNR", "Original SSIM", "Predict SSIM"])

        test_loss = 0
        with torch.no_grad():
            for i, (file_name, data, label) in enumerate(self.test_dataloader):
                torch.cuda.empty_cache()

                input, target = data.to(self.device), label.to(self.device)

                input_channel = input[:, 0:1, :, :]
                target_channel = target[:, 0:1, :, :]
                predict = self.model(input_channel)

                loss = self.criterion(predict, target_channel)
                test_loss += loss.item()

                _, _, height, width = input.shape
                resizer = Resize([height * 2, width * 2])
                input_exp = resizer(input)
                predict = torch.cat((predict, input_exp[:, 1:2, :, :], input_exp[:, 2:3, :, :]), 1)

                self.writer.add_scalar("Loss/test", loss.item(), i)

                # if i in test_idx:
                self.complete_test(i, input_exp, target, predict, file_name, csv_writer)

                if i % self.update_rate == 0:
                    pbar.update(min(self.update_rate, len(self.test_dataloader) - i))
                    pbar.set_postfix(Loss=f"{loss.item():.8f}")

        test_loss /= len(self.test_dataloader)
        pbar.set_postfix(Loss=f"{test_loss:.8f}")
        pbar.close()
        f.close()

    def complete_test(self, test_idx, input, target, predict, file_name, csv_writer):
        input = input.squeeze(0)*255
        target = target.squeeze(0)*255
        predict = predict.squeeze(0)*255

        input = YUV2RGB(input)
        predict = YUV2RGB(predict)
        target = YUV2RGB(target)
        
        input = input.clip(0, 255).cpu().detach().numpy().astype(np.uint8)
        target = target.clip(0, 255).cpu().detach().numpy().astype(np.uint8)
        predict = predict.clip(0, 255).cpu().detach().numpy().astype(np.uint8)

        height, width, _ = target.shape
        input = cv2.resize(input, (width, height), interpolation=cv2.INTER_CUBIC)

        p_psnr = np.round(PSNR(target, predict), 4)
        p_ssim = np.round(SSIM(target, predict, channel_axis=-1), 4)
        p_rmse = np.round(np.sqrt(MSE(target, predict)), 4)

        o_psnr = np.round(PSNR(target, input), 4)
        o_ssim = np.round(SSIM(target, input, channel_axis=-1), 4)
        o_rmse = np.round(np.sqrt(MSE(target, input)), 4)

        csv_writer.writerow([test_idx, o_rmse, p_rmse, o_psnr, p_psnr, o_ssim, p_ssim])

        images_dir = os.path.join(self.result_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        cv2.imwrite(os.path.join(images_dir, file_name[0]), cv2.cvtColor(predict, cv2.COLOR_RGB2BGR))

    def get_config(self, config_path):

        if config_path != "":
            with open(config_path, "r") as f:
                config = json.load(f)
            
            for key in config:
                if not hasattr(self, key):
                    warnings.warn(f"Warning: config has not attribute '{key}'")
                setattr(self, key, config[key])

    def save_model(self, file_name):
        f = os.path.join(self.models_path, file_name + ".pt")
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module.state_dict(), f)
        else:
            torch.save(self.model.state_dict(), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test'], help='train or test', required=True)
    parser.add_argument('--model', type=str, choices=['unet', 'rrdbnet'], required=True, help='unet or rrdbnet')
    parser.add_argument('--data_dir', type=str, default='dataset', help='path to the processed patch dataset')
    parser.add_argument('--test_data', default=('dataset_main/DIV2K_valid_HR', 'dataset_main/DIV2K_valid_LR*'), help='path to test directory')
    parser.add_argument('--result_dir', type=str, default='results', help='results directory')
    parser.add_argument('-c', '--config', type=str, help='config file with parameters')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--update_rate', type=int, default=10)
    parser.add_argument('--save_rate', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--do_validation', type=bool, default=True)
    parser.add_argument('--model_path', type=str, default='results/2022-12-06-20-39-21/models/best_val_model.pt', help='test only')
    parser.add_argument('--DEBUG', action='store_true')
    args = parser.parse_args()

    Orchestrator(args)