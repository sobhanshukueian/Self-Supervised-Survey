
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
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from utils import CosineAnnealingWarmupRestarts


import torch
import torchvision.utils
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.cuda import amp

from cifar_dataset import train_dataloader, train_val_dataloader, test_dataloader, vis_dataloader
from vis import show_batch
from configs import model_config
from utils import LARS, off_diagonal, get_color, get_colors, count_parameters, save, adjust_learning_rate, get_params_groups
from BYOL_model import BYOLNetwork
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


BATCH_SIZE = model_config['batch_size']
EPOCHS = model_config['EPOCHS']
device = model_config['device']
VERBOSE = model_config['VERBOSE']
SAVE_PLOTS = model_config['SAVE_PLOTS']
VISUALIZE_PLOTS = model_config['VISUALIZE_PLOTS']
SAVE_DIR = model_config['SAVE_DIR']
MODEL_NAME = model_config['MODEL_NAME']
WEIGHTS = model_config['WEIGHTS']
OPTIMIZER = model_config['OPTIMIZER']
EVALUATION_FREQ = model_config['EVALUATION_FREQ']
RESUME = model_config['RESUME']
RESUME_DIR = model_config["RESUME_DIR"]
USE_SCHEDULER = model_config["USE_SCHEDULER"]

class Trainer:
    # -----------------------------------------------INITIALIZE TRAINING-------------------------------------------------------------
    def __init__(self, device=device, epochs=EPOCHS, batch_size=BATCH_SIZE, save_dir=SAVE_DIR, train_loader=train_dataloader, train_val_loader=train_val_dataloader, valid_loader=test_dataloader, weights=WEIGHTS, verbose=VERBOSE, visualize_plots=VISUALIZE_PLOTS, save_plots=SAVE_PLOTS, model_name=MODEL_NAME, resume=RESUME, resume_dir=RESUME_DIR, use_scheduler = USE_SCHEDULER):
        self.device = device
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_name = model_name
        self.weights = weights
        self.visualize_plots = visualize_plots
        self.save_plots = save_plots
        # 0 == nothing || 1 == model architecture || 2 == print optimizer || 3 == model parameters
        self.verbose = verbose
        self.train_losses=[]
        self.val_losses=[]
        self.conf = {'Basic configs': model_config, 'Max_iter_num' : '', 'Epochs' : self.epochs, 'Trained_epoch' : 0, 'Optimizer' : '', 'Parameter_size' : '', "Model" : ''}
        self.ckpt = False
        self.resume = resume
        self.resume_dir = resume_dir
        self.start_epoch = 0
        self.use_scheduler = use_scheduler
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


        # get data loader
        self.train_loader, self.valid_loader, self.train_val_loader = train_loader, valid_loader, train_val_loader
        self.max_stepnum = len(self.train_loader)
        self.conf["Max_iter_num"] = self.max_stepnum


        # get model 
        self.model = BYOLNetwork().cuda()
        if self.verbose > 2:
            self.conf = count_parameters(self.model, self.conf)

        # Get optimizer
        self.optimizer = torch.optim.SGD(get_params_groups(self.model), lr=0.06, weight_decay=5e-4, momentum=0.9)


        if self.resume:
            self.start_epoch = self.ckpt["epoch"] + 1
            self.conf['resume'] += f" from epoch {self.start_epoch}"
        
        
        if self.use_scheduler:
            self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer,
                                            first_cycle_steps=50,
                                            cycle_mult=1.0,
                                            max_lr=model_config["LEARNING_RATE"],
                                            min_lr=5e-5,
                                            warmup_steps=30,
                                            gamma=0.8,
                                            last_epoch=self.start_epoch if self.resume else -1)
        # tensorboard
        
        # self.tblogger = SummaryWriter(self.save_dir) 

# -------------------------------------------------------------------------------TRAINING PROCESS-----------------------------------------------
    @staticmethod
    def prepro_data(batch_data, device, train):
        if train:
            images1, images2, targets = batch_data
            return images1.to(device), images2.to(device), targets.to(device)
        else:
            images1, targets = batch_data
            return images1.to(device), targets.to(device)

    # Each Train Step
    def train_step(self, batch_data):
        self.model.train()
        adjust_learning_rate(self.optimizer, self.epoch)

        image1, image2, targets = self.prepro_data(batch_data, self.device, True)
        
        preds, loss = self.model(image1, image2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.model.update_moving_average()

        return loss.cpu().detach().numpy()#, [pred.cpu().detach().numpy() for pred in preds]#, targets.cpu().detach().numpy()


    # Each Validation Step
    def val_step(self, batch_data):
        self.model.eval()
        image1, image2, targets = self.prepro_data(batch_data, self.device, True)

        # forward
        preds, loss = self.model(image1, image2)

        return loss.cpu().detach().numpy(), [pred.cpu().detach().numpy() for pred in preds], targets.cpu().detach().numpy()

    # Training Process
    def train(self):
        try:
            # training process prerequisite
            self.start_time = time.time()
            self.conf["Time"] = time.ctime(self.start_time)
            print('Start Training Process \nTime: {}'.format(time.ctime(self.start_time)))
            self.best_loss = np.inf
            knns = []

            # Epoch Loop
            for self.epoch in range(self.start_epoch, self.epochs):
                try:
                    self.conf["Trained_epoch"] = self.epoch
                    # ############################################################Train Loop
                    # Training loop
                    if self.epoch != 0:
                        pbar = enumerate(self.train_loader)
                        pbar = tqdm(pbar, desc=('%20s' * 3) % ('Phase' ,'Epoch', 'Total Loss'), total=self.max_stepnum)                        

                        for step, batch_data in pbar:
                            train_loss = self.train_step(batch_data)
                            self.train_losses.append(train_loss)
                            
                        print('%20s' * 3  % ("Train", f'{self.epoch}/{self.epochs}', train_loss.item()))                 
                        del pbar
            
                    # ############################################################Validation Loop

                    #     del vbar
                    if self.epoch % EVALUATION_FREQ == 0 : 
                        val_labels = []
                        val_embeddings = []

                        # Validation Loop
                        vbar = enumerate(self.valid_loader)
                        vbar = tqdm(vbar, desc=('%20s' * 3) % ('Phase' ,'Epoch', 'Total Loss'), total=len(self.valid_loader))
                        for step, batch_data in vbar:
                            self.val_loss, val_embeds, val_targets = self.val_step(batch_data)

                            if self.epoch != 0: self.val_losses.append(self.val_loss)

                            val_embeddings.extend(val_embeds[4])
                            val_labels.extend(val_targets)

                        print('%20s' * 3 % ("Validation", f'{self.epoch}/{self.epochs}', self.val_loss.item()))
                        del vbar

                        # PLot Losses
                        if self.epoch != 0: self.plot_loss()

                        # PLot Embeddings
                        self.plot_embeddings(np.array(val_embeddings), np.array(val_labels), 0)

                        knn_acc = knn_monitor(self.model.online, self.train_val_loader, self.valid_loader, self.epoch, k=200, hide_progress=False)

                        # # Delete Data after PLotting
                        del val_embeddings, val_labels
                        
                        if self.val_loss < self.best_loss:
                            self.best_loss=self.val_loss
                        
                        save(conf=self.conf, save_dir=self.save_dir, model_name=self.model_name, model=self.model, epoch=self.epoch, val_loss=self.val_loss, best_loss=self.best_loss, optimizer=self.optimizer)

                        print("\n---------------------------------------------------\n")

            
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
        COLS=3
        ROWS=1
        LINE_WIDTH = 2
        fig, ax = plt.subplots(ROWS, COLS, figsize=(COLS*10, ROWS*10))
        fig.suptitle("Losses Plot", fontsize=16)

        # train_mean_size = self.max_stepnum/self.batch_size
        ax[0].plot(np.arange(len(self.train_losses) / train_mean_size), np.mean(np.array(self.train_losses).reshape(-1, train_mean_size), axis=1), 'r',  label="training loss", linewidth=LINE_WIDTH)
        ax[0].set_title("Training Loss")

        val_mean_size = len(self.valid_loader)
        ax[1].plot(np.arange(len(self.val_losses) / val_mean_size), np.mean(np.array(self.val_losses).reshape(-1, val_mean_size), axis=1), 'g',  label="validation loss", linewidth=LINE_WIDTH)
        ax[1].set_title("Validation Loss")

        train_mean_size = self.max_stepnum
        ax[2].plot(np.arange(len(self.train_losses) / train_mean_size), np.mean(np.array(self.train_losses).reshape(-1, train_mean_size), axis=1), 'r',  label="training loss", linewidth=LINE_WIDTH)
        ax[2].plot(np.arange(len(self.val_losses) / val_mean_size), np.mean(np.array(self.val_losses).reshape(-1, val_mean_size), axis=1), 'g',  label="validation loss", linewidth=LINE_WIDTH)
        ax[2].set_title("Train Validation Loss")

        if self.save_plots:
            save_plot_dir = osp.join(self.save_dir, 'plots') 
            if not osp.exists(save_plot_dir):
                os.makedirs(save_plot_dir)
            plt.savefig("{}/epoch-{}-loss-plot.png".format(save_plot_dir, self.epoch)) 
        if self.visualize_plots:
            plt.show()

    def plot_embeddings(self, val_embeddings, val_labels, val_plot_size=0):
        if val_plot_size > 0:
            val_embeddings = np.array(val_embeddings[:val_plot_size])
            val_labels = np.array(val_labels[:val_plot_size])

        OUTPUT_EMBEDDING_SIZE = 10

        COLS = int(OUTPUT_EMBEDDING_SIZE / 2)
        ROWS = 1
        fig, ax = plt.subplots(ROWS, COLS, figsize=(COLS*10, ROWS*10))
        # fig.suptitle("Embeddings Plot", fontsize=16)
        for dim in range(0, OUTPUT_EMBEDDING_SIZE-1, 2):
            ax[int(dim/2)].set_title("Validation Samples for {} and {} dimensions".format(dim, dim+1))
            ax[int(dim/2)].scatter(val_embeddings[:, dim], val_embeddings[:, dim+1], c=get_colors(np.squeeze(val_labels)))
            
        if self.save_plots:
            save_plot_dir = osp.join(self.save_dir, 'plots') 
            if not osp.exists(save_plot_dir):
                os.makedirs(save_plot_dir)
            plt.savefig("{}/epoch-{}-plot.png".format(save_plot_dir, self.epoch)) 
        if self.visualize_plots:
            plt.show()

Trainer().train()

# Trainer(batch_size=32, device="cpu", epochs=50, verbose=0, weights="/content/runs/weights/best_SSL_epoch_45.pt").run("/content/data/faces/testing/s5/2.pgm", "/content/data/faces/testing/s7/4.pgm")