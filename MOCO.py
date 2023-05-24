
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
from utils import LARS, off_diagonal, get_color, get_colors, count_parameters, save, adjust_learning_rate, get_params_groups, adjust_loss_weights
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
        self.train_losses_s=[]
        self.val_losses=[]
        self.val_losses_s=[]
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
        self.model, self.conf, self.ckpt = get_model("MOCO", self.conf, self.resume, self.resume_dir, self.weights, self.verbose)
        self.model = self.model.to(device)
        
        self.model.init_var()
        self.model.target.requires_grad_(False)



        if self.verbose > 2:
            self.conf = count_parameters(self.model, self.conf)

        self.optimizer, self.conf = get_optimizer(get_params_groups(self.model), self.conf, self.resume, self.ckpt, optimizer=OPTIMIZER, lr0=model_config["LEARNING_RATE"], momentum=model_config["MOMENTUM"], weight_decay=model_config["WEIGHT_DECAY"], verbose=self.verbose)
        # self.optimizer = torch.optim.SGD(get_params_groups(self.model), lr=0.06, weight_decay=5e-4, momentum=0.9)

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

        image1, image2, targets = self.prepro_data(batch_data, self.device, True)
        
        _, loss, losses = self.model(image1, image2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.model.update_moving_average()

        return loss.cpu().detach().numpy(), [loss.cpu().detach().numpy() for loss in losses]#, [pred.cpu().detach().numpy() for pred in preds]#, targets.cpu().detach().numpy()


    # Each Validation Step
    def val_step(self, batch_data):
        self.model.eval()
        image1, image2, targets = self.prepro_data(batch_data, self.device, True)

        # forward
        preds, loss, losses = self.model(image1, image2)

        return loss.cpu().detach().numpy(), [pred.cpu().detach().numpy() for pred in preds], targets.cpu().detach().numpy(), [loss.cpu().detach().numpy() for loss in losses]

    # Training Process
    def train(self):
        try:
            # training process prerequisite
            self.start_time = time.time()
            self.conf["Time"] = time.ctime(self.start_time)
            print('Start Training Process \nTime: {}'.format(time.ctime(self.start_time)))
            self.best_loss = np.inf
            knns = []

            pf = '%10s' *2 + '%15s' * 4

            # Epoch Loop
            for self.epoch in range(self.start_epoch, self.epochs):
                try:
                    self.train_elosses = []
                    self.train_elosses_s = []

                    self.val_elosses = []
                    self.val_elosses_s = []

                    self.conf["Trained_epoch"] = self.epoch
                    # ############################################################Train Loop
                    # Training loop
                    # if self.epoch != 0:
                    adjust_learning_rate(self.optimizer, self.epoch, model_config["LEARNING_RATE"])
                    # self.loss_weight = adjust_loss_weights(self.epoch)
                    # print("Distance Loss Weight: ", self.loss_weight)


                    pbar = enumerate(self.train_loader)
                    pbar = tqdm(pbar, desc=pf % ('Phase' ,'Epoch', 'Total Loss', 'ISO KLD', 'KLD', 'Distance'), total=self.max_stepnum)                        

                    for step, batch_data in pbar:
                        train_loss, train_losses = self.train_step(batch_data)

                        if self.epoch != 0:
                            self.train_losses.append(train_loss)
                            self.train_losses_s.append(train_losses)

                        self.train_elosses.append(train_loss)
                        self.train_elosses_s.append(train_losses)

                        
                    print(pf  % ("Train", f'{self.epoch}/{self.epochs}', train_loss.mean(), train_losses[0].mean(), train_losses[1].mean(), train_losses[2].mean()))                 
                    del pbar
            
                    # ############################################################Validation Loop

                    if self.epoch % EVALUATION_FREQ == 0 : 
                        val_labels = []
                        val_embeddings = []
                        val_embeddings_normalized = []


                        # Validation Loop
                        vbar = enumerate(self.valid_loader)
                        vbar = tqdm(vbar, desc=pf % ('Phase' ,'Epoch', 'Total Loss', 'ISO KLD', 'KLD', 'Distance'), total=len(self.valid_loader))
                        for step, batch_data in vbar:
                            self.val_loss, val_embeds, val_targets, val_losses = self.val_step(batch_data)

                            if self.epoch != 0: 
                                self.val_losses.append(self.val_loss)
                                self.val_losses_s.append(val_losses)
                            self.val_elosses.append(self.val_loss)   
                            self.val_elosses_s.append(val_losses)

                            val_embeddings.extend(val_embeds[0])
                            val_embeddings_normalized.extend(val_embeds[1])
                            val_labels.extend(val_targets)

                        print(pf % ("Validation", f'{self.epoch}/{self.epochs}', self.val_loss.mean(), val_losses[0].mean(), val_losses[1].mean(), val_losses[2].mean()))
                        del vbar

                        # PLot Losses
                        if self.epoch != 0: 
                            self.plot_loss()
                        self.plot_eloss()

                        # PLot Embeddings
                        self.plot_embeddings(np.array(val_embeddings), np.array(val_labels), 0)
                        self.plot_embeddings(np.array(val_embeddings_normalized), np.array(val_labels), "Normalized Embeddings", 0)

                        knn_acc = knn_monitor(self.model.online, self.train_val_loader, self.valid_loader, self.epoch, k=200, hide_progress=False)

                        # Delete Data after PLotting
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
        COLS=5
        ROWS=2
        LINE_WIDTH = 2
        fig, ax = plt.subplots(ROWS, COLS, figsize=(COLS*10, ROWS*10))
        fig.suptitle("Losses Plot", fontsize=16)

        # train_mean_size = self.max_stepnum/self.batch_size
        ax[0, 0].plot(np.array(self.train_losses),  label="Training loss", linewidth=LINE_WIDTH+1)
        ax[0, 1].plot(np.array(self.train_losses_s)[:, 0], 'b--',  label="ISO KLD loss", linewidth=LINE_WIDTH-1)
        ax[0, 2].plot(np.array(self.train_losses_s)[:, 1], 'g--',  label="KLD loss", linewidth=LINE_WIDTH-1)
        ax[0, 3].plot(np.array(self.train_losses_s)[:, 2], 'r--',  label="Distance loss", linewidth=LINE_WIDTH-1)

        ax[0, 4].plot(np.array(self.train_losses),  label="Training loss", linewidth=LINE_WIDTH+1)
        ax[0, 4].plot(np.array(self.train_losses_s)[:, 0], 'b--',  label="ISO KLD loss", linewidth=LINE_WIDTH-1)
        ax[0, 4].plot(np.array(self.train_losses_s)[:, 1], 'g--',  label="KLD loss", linewidth=LINE_WIDTH-1)
        ax[0, 4].plot(np.array(self.train_losses_s)[:, 2], 'r--',  label="Distance loss", linewidth=LINE_WIDTH-1)

        ax[0, 0].set_title("Train Loss")
        ax[0, 1].set_title("ISO KLD loss")
        ax[0, 2].set_title("KLD loss")
        ax[0, 3].set_title("Distance loss")
        ax[0, 4].legend()

        # val_mean_size = len(self.valid_loader)
        ax[1, 0].plot(np.array(self.val_losses),  label="Validation loss", linewidth=LINE_WIDTH+1)
        ax[1, 1].plot(np.array(self.val_losses_s)[:, 0], 'b--',  label="ISO KLD loss", linewidth=LINE_WIDTH-1)
        ax[1, 2].plot(np.array(self.val_losses_s)[:, 1], 'g--',  label="KLD loss", linewidth=LINE_WIDTH-1)
        ax[1, 3].plot(np.array(self.val_losses_s)[:, 2], 'r--',  label="Distance loss", linewidth=LINE_WIDTH-1)

        ax[1, 4].plot(np.array(self.val_losses),  label="Validation loss", linewidth=LINE_WIDTH+1)
        ax[1, 4].plot(np.array(self.val_losses_s)[:, 0], 'b--',  label="ISO KLD loss", linewidth=LINE_WIDTH-1)
        ax[1, 4].plot(np.array(self.val_losses_s)[:, 1], 'g--',  label="KLD loss", linewidth=LINE_WIDTH-1)
        ax[1, 4].plot(np.array(self.val_losses_s)[:, 2], 'r--',  label="Distance loss", linewidth=LINE_WIDTH-1)
        
        ax[1, 0].set_title("Train Loss")
        ax[1, 1].set_title("ISO KLD loss")
        ax[1, 2].set_title("KLD loss")
        ax[1, 3].set_title("Distance loss")
        ax[1, 4].legend()

        if self.save_plots:
            save_plot_dir = osp.join(self.save_dir, 'plots') 
            if not osp.exists(save_plot_dir):
                os.makedirs(save_plot_dir)
            plt.savefig("{}/epoch-{}-loss-plot.png".format(save_plot_dir, self.epoch)) 
        if self.visualize_plots:
            plt.show()

    def plot_eloss(self, train_mean_size=1, val_mean_size=1):
        COLS=5
        ROWS=2
        LINE_WIDTH = 2
        fig, ax = plt.subplots(ROWS, COLS, figsize=(COLS*10, ROWS*10))
        fig.suptitle("Losses Plot", fontsize=16)

        # train_mean_size = self.max_stepnum/self.batch_size
        ax[0, 0].plot(np.array(self.train_elosses),  label="Training loss", linewidth=LINE_WIDTH+1)
        ax[0, 1].plot(np.array(self.train_elosses_s)[:, 0], 'b--',  label="ISO KLD loss", linewidth=LINE_WIDTH-1)
        ax[0, 2].plot(np.array(self.train_elosses_s)[:, 1], 'g--',  label="KLD loss", linewidth=LINE_WIDTH-1)
        ax[0, 3].plot(np.array(self.train_elosses_s)[:, 2], 'r--',  label="Distance loss", linewidth=LINE_WIDTH-1)

        ax[0, 4].plot(np.array(self.train_elosses),  label="Training loss", linewidth=LINE_WIDTH+1)
        ax[0, 4].plot(np.array(self.train_elosses_s)[:, 0], 'b--',  label="ISO KLD loss", linewidth=LINE_WIDTH-1)
        ax[0, 4].plot(np.array(self.train_elosses_s)[:, 1], 'g--',  label="KLD loss", linewidth=LINE_WIDTH-1)
        ax[0, 4].plot(np.array(self.train_elosses_s)[:, 2], 'r--',  label="Distance loss", linewidth=LINE_WIDTH-1)

        ax[0, 0].set_title("Train Loss")
        ax[0, 1].set_title("ISO KLD loss")
        ax[0, 2].set_title("KLD loss")
        ax[0, 3].set_title("Distance loss")
        ax[0, 4].legend()

        # val_mean_size = len(self.valid_loader)
        ax[1, 0].plot(np.array(self.val_elosses),  label="Validation loss", linewidth=LINE_WIDTH+1)
        ax[1, 1].plot(np.array(self.val_elosses_s)[:, 0], 'b--',  label="ISO KLD loss", linewidth=LINE_WIDTH-1)
        ax[1, 2].plot(np.array(self.val_elosses_s)[:, 1], 'g--',  label="KLD loss", linewidth=LINE_WIDTH-1)
        ax[1, 3].plot(np.array(self.val_elosses_s)[:, 2], 'r--',  label="Distance loss", linewidth=LINE_WIDTH-1)

        ax[1, 4].plot(np.array(self.val_elosses),  label="Validation loss", linewidth=LINE_WIDTH+1)
        ax[1, 4].plot(np.array(self.val_elosses_s)[:, 0], 'b--',  label="ISO KLD loss", linewidth=LINE_WIDTH-1)
        ax[1, 4].plot(np.array(self.val_elosses_s)[:, 1], 'g--',  label="KLD loss", linewidth=LINE_WIDTH-1)
        ax[1, 4].plot(np.array(self.val_elosses_s)[:, 2], 'r--',  label="Distance loss", linewidth=LINE_WIDTH-1)
        
        ax[1, 0].set_title("Train Loss")
        ax[1, 1].set_title("ISO KLD loss")
        ax[1, 2].set_title("KLD loss")
        ax[1, 3].set_title("Distance loss")
        ax[1, 4].legend()

        if self.save_plots:
            save_plot_dir = osp.join(self.save_dir, 'plots') 
            if not osp.exists(save_plot_dir):
                os.makedirs(save_plot_dir)
            plt.savefig("{}/epoch-{}-epoch-loss-plot.png".format(save_plot_dir, self.epoch)) 
        if self.visualize_plots:
            plt.show()

    def plot_embeddings(self, val_embeddings, val_labels, name="", val_plot_size=0):
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
            plt.savefig("{}/epoch-{}-{}-plot.png".format(save_plot_dir, self.epoch, name)) 
        if self.visualize_plots:
            plt.show()

Trainer().train()

# Trainer(batch_size=32, device="cpu", epochs=50, verbose=0, weights="/content/runs/weights/best_SSL_epoch_45.pt").run("/content/data/faces/testing/s5/2.pgm", "/content/data/faces/testing/s7/4.pgm")