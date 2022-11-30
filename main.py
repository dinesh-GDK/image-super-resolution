import os
import warnings
import argparse
import json
from datetime import datetime
import csv
import random

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanSquaredError

from dataset import DIV2K_dataset, convert_ycbcr_to_rgb
from utils.logger import get_logger
from models import UNet, SRCNN

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=False, help="config file with parameters")

class Orchestrator():

    mode = os.environ["MODE"]
    result_path = None
    use_gpu = True
    num_workers = 8
    update_rate = 10 # progress bar update per 10 iterations
    save_rate = 10
    # model_path = None
    model_path = "/home/dinesh/EE541_Project/Results/sample/models/best_train_model.pt"
    
    do_validation = False
    epoch = 500
    batch_size = 4
    lr = 1e-4
    weight_decay = 0

    train_input_path = "./sample_dataset/DIV2K_train_LR_bicubic_X2"
    train_output_path = "./sample_dataset/DIV2K_train_HR"

    val_input_path = "./sample_dataset/DIV2K_train_LR_bicubic_X2"
    val_output_path = "./sample_dataset/DIV2K_train_HR"

    test_input_path = "./sample_dataset/DIV2K_train_LR_bicubic_X2"
    test_output_path = "./sample_dataset/DIV2K_train_LR_bicubic_X2"
    
    DEBUG = True

    def __init__(self, config_path=None):
        
        if config_path is not None:
            self.get_config(config_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_gpu else "cpu")

        if self.DEBUG:
            self.result_path = os.path.join("Results", "sample")
        else:
            self.result_path = os.path.join("Results", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        self.models_path = os.path.join(self.result_path, "models")

        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(self.result_path, "tensorboard"))
        # log file
        self.logger = get_logger(os.path.join(self.result_path, "log"))

        self.model = UNet()
        self.criterion = torch.nn.MSELoss()

        if self.mode == "train":

            train_data = DIV2K_dataset(self.train_input_path, self.train_output_path)
            val_data   = DIV2K_dataset(self.val_input_path, self.val_output_path)
            self.train_dataloader = DataLoader(train_data, self.batch_size,
                                        shuffle=True, num_workers=self.num_workers)
            self.val_dataloader   = DataLoader(val_data, self.batch_size,
                                        shuffle=True, num_workers=self.num_workers)

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.model = torch.nn.DataParallel(self.model)
            self.train()

        elif self.mode == "test":

            test_data  = DIV2K_dataset(self.test_input_path, self.test_output_path, is_test=True)
            self.test_dataloader   = DataLoader(test_data, self.batch_size,
                                        shuffle=True, num_workers=self.num_workers)
            self.model = UNet().to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.test()

        else:
            self.logger.error("Invalid Mode")

    def train(self):

        self.logger.info("Start Training...")

        best_epoch_loss = float("inf")
        best_val_loss = float("inf")
        
        for epoch in range(self.epoch):

            torch.cuda.empty_cache()
            self.model.train()

            b="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
            pbar = tqdm(total=len(self.train_dataloader), bar_format=b,
                        desc=f"Epochs: {epoch+1}/{self.epoch}", ascii=' =')
            
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
            self.writer.add_scalar('Loss/epochs', epoch_loss, epoch)
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
                    desc=f"Validation", ascii=' -')
        
        val_loss = 0
        for i, (data, label) in enumerate(self.val_dataloader):

            input, target = data.to(self.device), label.to(self.device)
            
            predict = self.model(input)
            loss = self.criterion(predict, target)

            val_loss += loss.item()
            
            if i % self.update_rate == 0:
                pbar.update(min(self.update_rate, len(self.val_dataloader) - i))
                pbar.set_postfix(Loss=f"{loss.item():.8f}")
        
        val_loss /= len(self.val_dataloader)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        pbar.set_postfix(Loss=f"{val_loss:.8f}")
        pbar.close()

        return val_loss

    def test(self):

        self.logger.info("Start Testing...")

        torch.cuda.empty_cache()
        self.model.eval()

        test_idx = random.sample(range(0, len(self.test_dataloader)), min(1, 10))

        b="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
        pbar = tqdm(total=len(self.test_dataloader), bar_format=b,
                    desc=f"Test", ascii=" >")
        
        f = open(os.path.join(self.result_path, "_metrics.csv"), "w+")
        csv_writer = csv.writer(f)
        csv_writer.writerow(["S.No", "Original RMSE", "Predict RMSE", \
            "Original PSNR", "Predict PSNR", "Original SSIM", "Predict SSIM"])

        test_loss = 0
        for i, (data, label) in enumerate(self.test_dataloader):

            input, target = data.to(self.device), label.to(self.device)
            predict = self.model(input[:, 0:1, :, :])
            loss = self.criterion(predict, target[:, 0:1, :, :])
            test_loss += loss.item()

            predict = torch.cat((predict, input[:, 1:2, :, :], input[:, 2:3, :, :]), 1)

            self.writer.add_scalar("Loss/test", loss.item(), i)

            if i in test_idx:
                self.complete_test(i, input, target, predict, csv_writer)

            if i % self.update_rate == 0:
                pbar.update(min(self.update_rate, len(self.test_dataloader) - i))
                pbar.set_postfix(Loss=f"{loss.item():.8f}")
        
        test_loss /= len(self.test_dataloader)
        pbar.set_postfix(Loss=f"{test_loss:.8f}")
        pbar.close()
        f.close()

    def complete_test(self, test_idx, input, target, predict, csv_writer):

        input = input.squeeze(0)*255
        target = target.squeeze(0)*255
        predict = predict.squeeze(0)*255

        input = convert_ycbcr_to_rgb(input)
        predict = convert_ycbcr_to_rgb(predict)
        target = convert_ycbcr_to_rgb(target)
        
        input = input.clip(0, 255).cpu().detach().numpy().astype(np.uint8)
        target = target.clip(0, 255).cpu().detach().numpy().astype(np.uint8)
        predict = predict.clip(0, 255).cpu().detach().numpy().astype(np.uint8)

        p_psnr = np.round(PSNR(target, predict), 4)
        p_ssim = np.round(SSIM(target, predict, channel_axis=-1), 4)
        p_rmse = np.round(np.sqrt(MSE(target, predict)), 4)

        o_psnr = np.round(PSNR(target, input), 4)
        o_ssim = np.round(SSIM(target, input, channel_axis=-1), 4)
        o_rmse = np.round(np.sqrt(MSE(target, input)), 4)

        csv_writer.writerow([test_idx, o_rmse, p_rmse, o_psnr, p_psnr, o_ssim, p_ssim])

        images_path = os.path.join(self.result_path, "images")
        os.makedirs(images_path, exist_ok=True)

        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        ax[0].imshow(target)
        ax[1].imshow(predict)
        ax[0].set_axis_off()
        ax[1].set_axis_off()

        fig.suptitle(f"Random Image: {test_idx}")
        fig.savefig(os.path.join(images_path, f"{test_idx}.png"), bbox_inches="tight")
        plt.close()

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

    args = vars(parser.parse_args())
    Orchestrator(args["config"])
