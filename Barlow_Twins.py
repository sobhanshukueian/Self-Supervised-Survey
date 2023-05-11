
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
from utils import LARS, off_diagonal, get_color, get_colors, count_parameters, save
from Barlow_model import barlow_twins_loss, BarlowTwins
from main_utils import get_optimizer, get_model, eval_knn

def reproducibility(SEED):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
reproducibility(666)


if model_config["show_batch"]:
    show_batch(vis_dataloader)


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
KNN_EVALUATION_PERIOD = model_config['KNN_EVALUATION_PERIOD']

class Trainer:
    # -----------------------------------------------INITIALIZE TRAINING-------------------------------------------------------------
    def __init__(self, device=device, epochs=EPOCHS, batch_size=BATCH_SIZE, save_dir=SAVE_DIR, train_loader=train_dataloader, train_val_loader=train_val_dataloader, valid_loader=test_dataloader, weights=WEIGHTS, verbose=VERBOSE, visualize_plots=VISUALIZE_PLOTS, save_plots=SAVE_PLOTS, model_name=MODEL_NAME):
        self.device = device
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_ema = False
        self.model_name = model_name
        self.weights = weights
        self.visualize_plots = visualize_plots
        self.save_plots = save_plots
        # 0 == nothing || 1 == model architecture || 2 == print optimizer || 3 == model parameters
        self.verbose = verbose
        self.train_losses=[]
        self.val_losses=[]
        self.conf = {'Name' : self.model_name, 'Bacth_size' : self.batch_size, 'Max_iter_num' : '', 'Epochs' : self.epochs, 'Trained_epoch' : 0, 'Optimizer' : '', "Model" : '', 'Parameter_size' : ''}

        temm=0
        tmp_save_dir = self.save_dir
        while osp.exists(tmp_save_dir):
            tmp_save_dir = self.save_dir
            temm+=1
            tmp_save_dir += (str(temm))
        self.save_dir = tmp_save_dir
        print("Save Project in {} directory.".format(self.save_dir))
        del temm

        # get data loader
        self.train_loader, self.valid_loader, self.train_val_loader = train_loader, valid_loader, train_val_loader
        self.max_stepnum = len(self.train_loader)
        self.conf["Max_iter_num"] = self.max_stepnum
        
        # get model 
        self.model, self.conf = get_model("barlow", self.conf, self.weights, self.device, self.verbose)
        if self.verbose > 2:
            self.conf = count_parameters(self.model, self.conf)

        # Get optimizer
        self.optimizer, self.conf = get_optimizer(self.model.parameters(), self.conf, optimizer=OPTIMIZER, lr0=0.001, momentum=0.937, verbose=self.verbose)
    
        # tensorboard
        # self.tblogger = SummaryWriter(self.save_dir)  

# -------------------------------------------------------------------------------TRAINING PROCESS-----------------------------------------------
    @staticmethod
    def prepro_data(batch_data, device):
        images1 = batch_data[0].to(device)
        images2 = batch_data[1].to(device) 
        targets = batch_data[2].to(device)
        # images = torch.cat([images1, images2], dim=0)
        # targets = torch.cat([targets, targets], dim=0)

        return images1, images2, targets

    # Each Train Step
    def train_step(self, batch_data, step):
        images1, images2, targets = self.prepro_data(batch_data, self.device)
        # forward
        # print(images.shape)
        with amp.autocast(enabled=self.device != 'cpu'):
            # print("\n#############################################\n")
            # print(images)
            # print("\n#######################################\n")


            # COLS = len(images)
            # ROWS = 1
            # fig, ax = plt.subplots(ROWS, COLS, figsize=(COLS*10, ROWS*10))
            # # fig.suptitle("Embeddings Plot", fontsize=16)
            # for indx, img in enumerate(images):
            #     ax[indx].imshow(img.view(3, 100, 100).permute(1 , 2 , 0), interpolation='nearest')
            # plt.show()

            pred1, pred2 = self.model(images1, images2)

            # print("\n=====================================================\n")
            # print(preds)
            # print("\n=====================================================\n")
            loss = barlow_twins_loss(pred1, pred2)

        # backward
        # print("\n**********************************************\n")
        # print(loss)
        # print("\n**********************************************\n")

        # print("\n1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111\n")
        # for name, param in self.model.named_parameters():
        #     if torch.isnan(param).any():
        #         print("NaN values found in parameter:", name)
        # print("\n1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111\n")
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # print("\n2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222\n")
        # for name, param in self.model.named_parameters():
        #     if torch.isnan(param).any():
        #         print("NaN values found in parameter:", name)
        # print("\n2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222\n")

        return loss.cpu().detach().numpy(), pred1.cpu().detach().numpy(), pred2.cpu().detach().numpy()


    # Each Validation Step
    def val_step(self, batch_data):
        self.model.eval()
        images1, images2, targets = self.prepro_data(batch_data, self.device)

        # forward
        pred1 = self.model(images1)
        pred2 = self.model(images2)
        loss = barlow_twins_loss(pred1, pred2)

        return loss.cpu().detach().numpy(), pred1.cpu().detach().numpy(), pred2.cpu().detach().numpy(), targets.cpu().detach().numpy()

    # Training Process
    def train(self):
        try:
            # training process prerequisite
            self.start_time = time.time()
            print('Start Training Process \nTime: {}'.format(time.ctime(self.start_time)))
            self.scaler = amp.GradScaler(enabled=self.device != 'cpu')
            self.best_loss = np.inf
            knns=[]

            # Epoch Loop
            for self.epoch in range(0, self.epochs):
                try:
                    self.conf["Trained_epoch"] = self.epoch


                    # ############################################################Train Loop
                    if self.epoch != 0 :
                        # Training loop
                        self.model.train(True)
                        pbar = enumerate(self.train_loader)
                        pbar = tqdm(pbar, desc=('%20s' * 3) % ('Phase' ,'Epoch', 'Total Loss'), total=self.max_stepnum)
                        for step, batch_data in pbar:
                            self.train_loss, embed1, embed2 = self.train_step(batch_data, step)
                            self.train_losses.append(self.train_loss)        
                            # pbar.set_description(f"Epoch: {self.epoch}/{self.epochs}\tTrain Loss: {self.train_loss}")
                            pf = '%20s' * 3 # print format
                        print(pf % ("Train", f'{self.epoch}/{self.epochs}', self.train_loss.item()))
                        del pbar


                    # ############################################################Validation Loop

                    labels = []
                    embeddings = []
                    val_labels = []
                    val_embeddings = []

                    if self.epoch % KNN_EVALUATION_PERIOD == 0:
                        # Train Validation Loop
                        vbar = enumerate(self.train_val_loader)
                        vbar = tqdm(vbar, desc=('%20s' * 3) % ('Phase' ,'Epoch', 'Total Loss'), total=len(self.train_val_loader))
                        for step, batch_data in vbar:
                            self.val_loss, val_embed1, val_embed2, val_targets = self.val_step(batch_data)
                            # vbar.set_description(f"Epoch: {self.epoch}/{self.epochs}\tTrain Validation Loss: {self.val_loss}")
                            embeddings.extend(val_embed1)
                            # embeddings.extend(val_embed2)
                            labels.extend(val_targets)
                            # labels.extend(val_targets)
                            pf = '%20s' * 3 # print format
                        print(pf % ("Train Validation", f'{self.epoch}/{self.epochs}', self.val_loss.item()))
                        del vbar


                    # Validation Loop
                    vbar = enumerate(self.valid_loader)
                    vbar = tqdm(vbar, desc=('%20s' * 3) % ('Phase' ,'Epoch', 'Total Loss'), total=len(self.valid_loader))
                    for step, batch_data in vbar:
                        self.val_loss, val_embed1, val_embed2, val_targets = self.val_step(batch_data)
                        self.val_losses.append(self.val_loss)
                        # vbar.set_description(f"Epoch: {self.epoch}/{self.epochs}\tValidation Loss: {self.val_loss}")
                        val_embeddings.extend(val_embed1)
                        # val_embeddings.extend(val_embed2)
                        val_labels.extend(val_targets)
                        # val_labels.extend(val_targets)
                        pf = '%20s' * 3 # print format
                    print(pf % ("Validation", f'{self.epoch}/{self.epochs}', self.val_loss.item()))
                    del vbar

                    # PLot Losses
                    if self.epoch != 0: self.plot_loss()

                    # PLot Embeddings
                    # plot_size = BATCH_SIZE
                    self.plot_embeddings(np.array(val_embeddings), np.array(val_labels), 0)

                    if self.epoch % KNN_EVALUATION_PERIOD == 0 : 
                        knn_acc = eval_knn(embeddings, labels, knns)
                        knns.append(knn_acc)

                    # # Delete Data after PLotting
                    del val_embeddings, val_labels, embeddings, labels

                    
                    if self.val_loss < self.best_loss:
                        self.best_loss=self.val_loss
            
                except Exception as _:
                    print('ERROR in training steps.')
                    raise
                try:
                    save(conf=self.conf, save_dir=self.save_dir, model_name=self.model_name, model=self.model, epoch=self.epoch, val_loss=self.val_loss, best_loss=self.best_loss)
                except Exception as _:
                    print('ERROR in evaluate and save model.')
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