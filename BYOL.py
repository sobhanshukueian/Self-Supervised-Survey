
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
from utils import LARS, off_diagonal, get_color, get_colors, count_parameters, save
from BYOL_model import BYOLNetwork, byol_loss
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
        self.use_ema = False
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
        self.model, self.conf, self.ckpt = get_model("byol", self.conf, self.resume, self.resume_dir, self.weights, self.verbose)
        self.model = self.model.to(device)
        if self.verbose > 2:
            self.conf = count_parameters(self.model, self.conf)

        # Get optimizer
        self.optimizer, self.conf = get_optimizer(self.model.parameters(), self.conf, self.resume, self.ckpt, optimizer=OPTIMIZER, lr0=model_config["LEARNING_RATE"], momentum=model_config["MOMENTUM"], weight_decay=model_config["WEIGHT_DECAY"], verbose=self.verbose)


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
    def prepro_data(batch_data, device):
        images1 = batch_data[0].to(device)
        images2 = batch_data[1].to(device) 
        targets = batch_data[2].to(device)
        return images1, images2, targets

    # Each Train Step
    def train_step(self, batch_data):
        image1, image2, targets = self.prepro_data(batch_data, self.device)
        with amp.autocast(enabled=self.device != 'cpu'):
            preds = self.model(image1, image2)
            loss = (byol_loss(preds[0], preds[2]) + byol_loss(preds[1], preds[3])).mean()


        # state_dict_before = copy.deepcopy(self.model.state_dict())

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        



        # state_dict_after = copy.deepcopy(self.model.state_dict())
        # # Compare the two state_dicts
        # for name, param in state_dict_before.items():
        #     if torch.all(torch.eq(param, state_dict_after[name])):
        #         continue
        #     else:
        #         print(f"{name} has changed.")


        # Update Target 
        self.model.update_moving_average()

        # state_dict_afterer = copy.deepcopy(self.model.state_dict())
        # for name, param in state_dict_after.items():
        #     if torch.all(torch.eq(param, state_dict_afterer[name])):
        #         continue
        #     else:
        #         print(f"{name} has changed.After Moving Average ")


        return loss.cpu().detach().numpy()#, [pred.cpu().detach().numpy() for pred in preds]#, targets.cpu().detach().numpy()


    # Each Validation Step
    def val_step(self, batch_data):
        self.model.eval()
        image1, image2, targets = self.prepro_data(batch_data, self.device)

        # forward
        preds = self.model(image1, image2)
        val_pred = self.model(image1)
        loss = (byol_loss(preds[0], preds[2]) + byol_loss(preds[1], preds[3])).mean()
        return loss.cpu().detach().numpy(), [pred.cpu().detach().numpy() for pred in preds], targets.cpu().detach().numpy(), val_pred.cpu().detach().numpy()

    # Training Process
    def train(self):
        try:
            # training process prerequisite
            self.start_time = time.time()
            print('Start Training Process \nTime: {}'.format(time.ctime(self.start_time)))
            self.scaler = amp.GradScaler(enabled=self.device != 'cpu')
            self.best_loss = np.inf
            knns = []

            # Epoch Loop
            for self.epoch in range(self.start_epoch, self.epochs):
                try:
                    self.conf["Trained_epoch"] = self.epoch

                    # ############################################################Train Loop
                    if self.epoch != 0:
                        # Training loop
                        self.model.train(True)
                        pbar = enumerate(self.train_loader)
                        # pbar = tqdm(pbar, total=self.max_stepnum)
                        pbar = tqdm(pbar, desc=('%20s' * 3) % ('Phase' ,'Epoch', 'Total Loss'), total=self.max_stepnum)                        
                        for step, batch_data in pbar:
                            self.train_loss = self.train_step(batch_data)
                            self.train_losses.append(self.train_loss)
                            # pbar.set_description(f"Epoch: {self.epoch}/{self.epochs}\tTrain Loss: {self.train_loss}  ")                 
                            pf = '%20s' * 3 # print format
                        print(pf % ("Train", f'{self.epoch}/{self.epochs}', self.train_loss.item()))                 
                        del pbar
                    
                        if self.scheduler: 
                            self.scheduler.step()
                            print("Learning Rate : ", optimizer.state_dict()['param_groups'][0]['lr'])

                    # ############################################################Validation Loop

                    labels = []
                    embeddings = []
                    val_labels = []
                    val_embeddings = []
                    


                    if self.epoch % KNN_EVALUATION_PERIOD == 0:
                        # Train Validation Loop
                        vbar = enumerate(self.train_val_loader)
                        # vbar = tqdm(vbar, total=len(self.train_val_loader))
                        vbar = tqdm(vbar, desc=('%20s' * 3) % ('Phase' ,'Epoch', 'Total Loss'), total=len(self.train_val_loader))

                        for step, batch_data in vbar:
                            self.val_loss, val_embedss, val_targets, val_embeds = self.val_step(batch_data)
                            # vbar.set_description(f"Epoch: {self.epoch}/{self.epochs}\tTrain Validation Loss: {self.val_loss}  ")
                            embeddings.extend(val_embeds)
                            labels.extend(val_targets)
                            pf = '%20s' * 3 # print format
                        print(pf % ("Train Validation", f'{self.epoch}/{self.epochs}', self.val_loss.item()))
                       
                        del vbar

                    # Validation Loop
                    vbar = enumerate(self.valid_loader)
                    # vbar = tqdm(vbar, total=len(self.valid_loader))
                    vbar = tqdm(vbar, desc=('%20s' * 3) % ('Phase' ,'Epoch', 'Total Loss'), total=len(self.valid_loader))
                    for step, batch_data in vbar:
                        self.val_loss, val_embedss, val_targets, val_embeds = self.val_step(batch_data)
                        if self.epoch != 0: self.val_losses.append(self.val_loss)
                        # vbar.set_description(f"Epoch: {self.epoch}/{self.epochs}\tValidation Loss: {self.val_loss}  ")
                        val_embeddings.extend(val_embeds)
                        val_labels.extend(val_targets)
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

                    print("\n-----------------------------------------------\n")

            
                except Exception as _:
                    print('ERROR in training steps.')
                    raise
                try:
                    save(conf=self.conf, save_dir=self.save_dir, model_name=self.model_name, model=self.model, epoch=self.epoch, val_loss=self.val_loss, best_loss=self.best_loss, optimizer=self.optimizer)
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