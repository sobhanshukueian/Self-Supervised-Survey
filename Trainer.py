
import os
import numpy as np
import time
import math
from copy import deepcopy
import os.path as osp
import shutil
from prettytable import PrettyTable
import json
from sklearn.metrics import auc
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from utils import CosineAnnealingWarmupRestarts, set_logging


import torch
import torchvision.utils
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.cuda import amp

from stl_dataset import get_stl_data
from cifar_dataset import get_cifar_data
from vis import plot_embeddings
from configs import model_config
from utils import LARS, off_diagonal, get_color, get_colors, count_parameters, save, adjust_learning_rate, get_params_groups
# from BYOL_model import BYOLNetwork
from main_utils import get_optimizer, get_model
from knn_eval import knn_monitor


def reproducibility(SEED):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
reproducibility(666)


if model_config["dataset"] == "STL10":
    train_dataloader, train_val_dataloader, test_dataloader, vis_dataloader = get_stl_data()
else:
    train_dataloader, train_val_dataloader, test_dataloader, vis_dataloader = get_cifar_data()


class Trainer:
    # -----------------------------------------------INITIALIZE TRAINING-------------------------------------------------------------
    def __init__(self, train_loader=train_dataloader, train_val_loader=train_val_dataloader, valid_loader=test_dataloader):
        self.device = model_config['device']
        self.save_dir = model_config['SAVE_DIR']
        self.batch_size = model_config['batch_size']
        self.epochs = model_config['EPOCHS']
        self.model_name = model_config['MODEL_NAME']
        self.weights = model_config['WEIGHTS']
        self.visualize_plots = model_config["VISUALIZE_PLOTS"]
        self.save_plots = model_config["SAVE_PLOTS"]
        # 0 == nothing || 1 == model architecture || 2 == print optimizer || 3 == model parameters
        self.verbose = model_config["VERBOSE"]
        self.train_losses=[]
        self.train_losses_s=[]
        self.val_losses=[]
        self.val_losses_s=[]
        self.conf = {'Basic configs': model_config, 'Max_iter_num' : '', 'Epochs' : self.epochs, 'Trained_epoch' : 0, 'Optimizer' : '', 'Parameter_size' : '', "Model" : ''}
        self.ckpt = False
        self.resume = model_config['RESUME']
        self.resume_dir = model_config["RESUME_DIR"]
        self.start_epoch = 0
        self.use_scheduler = model_config["USE_SCHEDULER"]
        self.scheduler = False
        

        temm=0
        tmp_save_dir = self.save_dir
        while osp.exists(tmp_save_dir):
            tmp_save_dir = self.save_dir
            temm+=1
            tmp_save_dir += (str(temm))
        self.save_dir = tmp_save_dir
        del temm
        print("Save Project in {} directory.".format(self.save_dir))

        self.logger = set_logging(self.save_dir, self.model_name)

        # get data loader
        self.train_loader, self.valid_loader, self.train_val_loader = train_loader, valid_loader, train_val_loader
        self.max_stepnum = len(self.train_loader)
        self.conf["Max_iter_num"] = self.max_stepnum


        # get model 
        self.model, self.conf, self.ckpt = get_model(model_config["MODEL_NAME"], self.conf, self.resume, self.resume_dir, self.weights, self.verbose)
        self.model = self.model.to(self.device)

        # if self.verbose > 2:
        self.conf = count_parameters(self.logger, self.model, self.conf)

        self.optimizer, self.conf = get_optimizer(self.logger, get_params_groups(self.model), self.conf, self.resume, self.ckpt, model_config['OPTIMIZER'], lr0=model_config["LEARNING_RATE"], momentum=model_config["MOMENTUM"], weight_decay=model_config["WEIGHT_DECAY"], verbose=self.verbose)
        # self.optimizer = torch.optim.SGD(get_params_groups(self.model), lr=0.06, weight_decay=5e-4, momentum=0.9)

        if self.resume:
            self.start_epoch = self.ckpt["epoch"] + 1
            self.conf['resume'] += f" from epoch {self.start_epoch}"
        

# -------------------------------------------------------------------------------TRAINING PROCESS-----------------------------------------------
    @staticmethod
    def prepro_data(batch_data, device):
        images1, images2, targets = batch_data
        return images1.to(device), images2.to(device), targets


    # Each Train Step
    def train_step(self, batch_data):
        image1, image2, targets = self.prepro_data(batch_data, self.device)
        
        preds, loss, losses = self.model(image1, image2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), [loss.item() for loss in losses]

    def train(self):
        self.model.train()
        adjust_learning_rate(self.optimizer, self.epoch, model_config["LEARNING_RATE"])

        pbar = enumerate(self.train_loader)
        pbar = tqdm(pbar, desc=('%20s' * 3) % ('Phase' ,'Epoch', 'Total Loss'), total=self.max_stepnum)                        
        self.logger.warning(('%20s' * 3) % ('Phase' ,'Epoch', 'Total Loss'))
        for step, batch_data in pbar:
            train_loss, train_losses = self.train_step(batch_data)
            if self.epoch != 0: 
                self.train_losses.append(train_loss)
                self.train_losses_s.append(train_losses)
            
        print('%20s' * 3  % ("Train", f'{self.epoch}/{self.epochs}', train_loss.item()))     
        self.logger.warning('%20s' * 3  % ("Train", f'{self.epoch}/{self.epochs}', train_loss.item()))                 

        del pbar


    # Each Validation Step
    def val_step(self, batch_data):
        self.model.eval()
        image1, image2, targets = self.prepro_data(batch_data, self.device)
        preds, loss, losses = self.model(image1, image2)
        return loss.item(), [pred.cpu().detach().numpy() for pred in preds], targets, [loss.item() for loss in losses]

    def validation(self):
        val_labels = []
        val_embeddings = []

        # Validation Loop
        vbar = enumerate(self.valid_loader)
        vbar = tqdm(vbar, desc=('%20s' * 3) % ('Phase' ,'Epoch', 'Total Loss'), total=len(self.valid_loader))
        self.logger.warning(('%20s' * 3) % ('Phase' ,'Epoch', 'Total Loss'))

        for step, batch_data in vbar:
            val_loss, val_embeds, val_targets, val_losses = self.val_step(batch_data)

            if self.epoch != 0: 
                self.val_losses.append(val_loss)
                self.val_losses_s.append(val_losses)

            val_embeddings.extend(val_embeds[0])
            val_labels.extend(val_targets)

        print('%20s' * 3 % ("Validation", f'{self.epoch}/{self.epochs}', val_loss))
        self.logger.warning('%20s' * 3 % ("Validation", f'{self.epoch}/{self.epochs}', val_loss))
        
        # PLot Losses
        if self.epoch != 0: self.plot_loss()
        
        # PLot Embeddings
        plot_embeddings(self.epoch, np.array(val_embeddings), np.array(val_labels), 0)

        if val_loss < self.best_loss:
            self.best_loss= val_loss

    def knn_eval(self):
        validation_model = deepcopy(self.model.encoder_q)
        # if validation_model.fc : 
        #     validation_model.fc = nn.Identity()
        knn_acc = knn_monitor(self.logger, validation_model, self.train_val_loader, self.valid_loader, self.epoch, k=200, hide_progress=False)
        results['KNN_acc@1'].append(knn_acc)
        
        filename = self..save_dir + "/KNN.csv"
        
        file_exists = os.path.isfile(filename)

        # Open the CSV file in append mode
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)

            # If the file doesn't exist or doesn't contain the header row, add it
            if not file_exists or 'epoch,KNN_acc' not in open(filename).read():
                writer.writerow(['epoch', 'KNN_acc'])

            # Write a new row with epoch and KNN_acc values
            writer.writerow([self.epoch, knn_acc])


    # Training Process
    def train(self):
        try:
            # training process prerequisite
            self.start_time = time.time()
            self.conf["Time"] = time.ctime(self.start_time)
            print('Start Training Process \nTime: {}'.format(time.ctime(self.start_time)))
            self.logger.warning('Start Training Process \nTime: {}'.format(time.ctime(self.start_time)))

            self.best_loss = np.inf

            # Epoch Loop
            for self.epoch in range(self.start_epoch, self.epochs+1):
                try:
                    self.conf["Trained_epoch"] = self.epoch

                    # ############################################################Train Loop
                    if self.epoch != 0:
                        self.train()
                    
                    self.knn_eval()

                    # ###########################################################Validation Loop
                    if self.epoch % model_config['VALIDATION_FREQ'] == 0 : 
                        self.validation()
                        
                    save(conf=self.conf, save_dir=self.save_dir, model_name=self.model_name, model=self.model, epoch=self.epoch, val_loss=self.val_loss, best_loss=self.best_loss, optimizer=self.optimizer)
         
                except Exception as _:
                    print('ERROR in training steps.')
                    raise

        except Exception as _:
            print('ERROR in training loop or eval/save model.')
            raise
        finally:
            finish_time = time.time()
            print(f'\nTraining completed in {time.ctime(finish_time)} \nIts Done in: {(time.time() - self.start_time) / 3600:.3f} hours.') 



    # -------------------------------------------------------Training Callback after each epoch--------------------------
    def plot_loss(self, train_mean_size=1, val_mean_size=1):
        FIRST_ = "Contrastive Loss1"
        SECOND_ = "Contrastive Loss2"
        THIRD_ = "ISOKLD LOSS"

        COLS=5
        ROWS=2
        LINE_WIDTH = 2
        fig, ax = plt.subplots(ROWS, COLS, figsize=(COLS*10, ROWS*10))
        fig.suptitle("Losses Plot", fontsize=16)

        # train_mean_size = self.max_stepnum/self.batch_size
        ax[0, 0].plot(np.array(self.train_losses),  label="Training loss", linewidth=LINE_WIDTH+1)
        ax[0, 1].plot(np.array(self.train_losses_s)[:, 0], 'b--',  label=FIRST_, linewidth=LINE_WIDTH-1)
        ax[0, 2].plot(np.array(self.train_losses_s)[:, 1], 'g--',  label=SECOND_, linewidth=LINE_WIDTH-1)
        ax[0, 3].plot(np.array(self.train_losses_s)[:, 2], 'r--',  label=THIRD_, linewidth=LINE_WIDTH-1)

        ax[0, 4].plot(np.array(self.train_losses),  label="Training loss", linewidth=LINE_WIDTH+1)
        ax[0, 4].plot(np.array(self.train_losses_s)[:, 0], 'b--',  label=FIRST_, linewidth=LINE_WIDTH-1)
        ax[0, 4].plot(np.array(self.train_losses_s)[:, 1], 'g--',  label=SECOND_, linewidth=LINE_WIDTH-1)
        ax[0, 4].plot(np.array(self.train_losses_s)[:, 2], 'r--',  label=THIRD_, linewidth=LINE_WIDTH-1)

        ax[0, 0].set_title("Train Loss")
        ax[0, 1].set_title(FIRST_)
        ax[0, 2].set_title(SECOND_)
        ax[0, 3].set_title(THIRD_)
        ax[0, 4].legend()

        # val_mean_size = len(self.valid_loader)
        ax[1, 0].plot(np.array(self.val_losses),  label="Validation loss", linewidth=LINE_WIDTH+1)
        ax[1, 1].plot(np.array(self.val_losses_s)[:, 0], 'b--',  label=FIRST_, linewidth=LINE_WIDTH-1)
        ax[1, 2].plot(np.array(self.val_losses_s)[:, 1], 'g--',  label=SECOND_, linewidth=LINE_WIDTH-1)
        ax[1, 3].plot(np.array(self.val_losses_s)[:, 2], 'r--',  label=THIRD_, linewidth=LINE_WIDTH-1)

        ax[1, 4].plot(np.array(self.val_losses),  label="Validation loss", linewidth=LINE_WIDTH+1)
        ax[1, 4].plot(np.array(self.val_losses_s)[:, 0], 'b--',  label=FIRST_, linewidth=LINE_WIDTH-1)
        ax[1, 4].plot(np.array(self.val_losses_s)[:, 1], 'g--',  label=SECOND_, linewidth=LINE_WIDTH-1)
        ax[1, 4].plot(np.array(self.val_losses_s)[:, 2], 'r--',  label=THIRD_, linewidth=LINE_WIDTH-1)
        
        ax[1, 0].set_title("Train Loss")
        ax[1, 1].set_title(FIRST_)
        ax[1, 2].set_title(SECOND_)
        ax[1, 3].set_title(THIRD_)
        ax[1, 4].legend()

        if self.save_plots:
            save_plot_dir = osp.join(self.save_dir, 'plots') 
            if not osp.exists(save_plot_dir):
                os.makedirs(save_plot_dir)
            plt.savefig("{}/epoch-{}-loss-plot.png".format(save_plot_dir, self.epoch)) 
        if self.visualize_plots:
            plt.show()

Trainer().train()

# Trainer(batch_size=32, device="cpu", epochs=50, verbose=0, weights="/content/runs/weights/best_SSL_epoch_45.pt").run("/content/data/faces/testing/s5/2.pgm", "/content/data/faces/testing/s7/4.pgm")