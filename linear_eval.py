
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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


from cifar_dataset import train_dataloader, train_val_dataloader, test_dataloader
from vis import show_batch
from configs import model_config
from utils import LARS, off_diagonal, get_color, get_colors, get_params_groups, LinearClassifier, accuracy, compute_acc, save
from BYOL_model import BYOLNetwork
from Barlow_model import BarlowTwins
from main_utils import get_optimizer, get_model, eval_knn

MODE = "byol" #@param
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


class Linear_Validator:
    # -----------------------------------------------INITIALIZE TRAINING-------------------------------------------------------------
    def __init__(self, device=device, epochs=EPOCHS, batch_size=BATCH_SIZE, save_dir=SAVE_DIR, train_loader=train_dataloader, valid_loader=test_dataloader, weights=WEIGHTS, verbose=VERBOSE, visualize_plots=VISUALIZE_PLOTS, save_plots=SAVE_PLOTS, model_name=MODEL_NAME, resume=RESUME, resume_dir=RESUME_DIR, use_scheduler = USE_SCHEDULER, mode=MODE):
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
        self.mode = mode
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
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.max_stepnum = len(self.train_loader)
        self.conf["Max_iter_num"] = self.max_stepnum
        
        # get model 
        self.model, self.conf, self.ckpt = get_model(self.mode, self.conf, self.resume, self.resume_dir, self.weights, self.verbose)
        self.model = self.model.to(device)
        self.model.eval()

        self.linear_classifier = LinearClassifier(model_config["EMBEDDING_SIZE"])
        self.linear_classifier = self.linear_classifier.to(device)

        if self.verbose > 2:
            self.count_parameters()

        # Get optimizer
        self.optimizer, self.conf = get_optimizer(self.linear_classifier.parameters(), self.conf, self.resume, self.ckpt, optimizer=OPTIMIZER, lr0=model_config["LEARNING_RATE"], momentum=model_config["MOMENTUM"], weight_decay=model_config["WEIGHT_DECAY"], verbose=self.verbose)
        
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
        
    
        # tensorboard
        # self.tblogger = SummaryWriter(self.save_dir) 

# -------------------"------------------------------------------------------------TRAINING PROCESS-----------------------------------------------
    @staticmethod
    def prepro_data(batch_data, device):
        inputs, _, labels = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device)
        labels = labels.view(-1)
        return inputs.to(device), labels.to(device)

    # Each Train Step
    def train_step(self, batch_data):
        inputs, labels = self.prepro_data(batch_data, self.device)
        with torch.no_grad():
            outputs = self.model(inputs)

        outputs = self.linear_classifier(outputs)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.cpu().detach().numpy(), outputs.cpu().detach(), labels.cpu().detach()


    # Each Validation Step
    def val_step(self, batch_data):
        self.linear_classifier.eval()

        inputs, labels = self.prepro_data(batch_data, self.device)
        outputs = self.model(inputs)
        outputs = self.linear_classifier(outputs)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)

        return loss.cpu().detach().numpy(), outputs.cpu().detach(), labels.cpu().detach()

    # Training Process
    def run(self):
        try:
            torch.cuda.empty_cache()

            # training process prerequisite
            self.start_time = time.time()
            self.conf["Time"] = time.ctime(self.start_time)
            print('Start Training Process \nTime: {}'.format(time.ctime(self.start_time)))
            self.scaler = amp.GradScaler(enabled=self.device != 'cpu')
            self.best_loss = np.inf

            # Epoch Loop
            for self.epoch in range(0, self.epochs):
                try:
                    self.conf["Trained_epoch"] = self.epoch

                    train_predictions = torch.tensor([])
                    train_labels = torch.tensor([])

                    validation_predictions = torch.tensor([])
                    validation_labels = torch.tensor([])

                    # ############################################################Train Loop
                    # Training loop
                    self.model.train(False)
                    self.linear_classifier.train(True)

                    pbar = enumerate(self.train_loader)
                    pbar = tqdm(pbar, desc=('%20s' * 3) % ('Phase' ,'Epoch', 'Total Loss'), total=self.max_stepnum)
                    for step, batch_data in pbar:
                        self.train_loss, predictions, labels = self.train_step(batch_data)
                        
                        train_predictions = torch.cat([train_predictions, predictions], dim=0)
                        train_labels = torch.cat([train_labels, labels], dim=0)
                        
                        self.train_losses.append(self.train_loss)
                        pf = '%20s' * 3 # print format
                    print(pf % ("Train", f'{self.epoch}/{self.epochs}', self.train_loss.item()))                 
                    del pbar

                    if self.scheduler: 
                        self.scheduler.step()
                    print("Learning Rate : ", self.optimizer.state_dict()['param_groups'][0]['lr'])


                    # ############################################################Validation Loop

                    # Validation Loop
                    vbar = enumerate(self.valid_loader)
                    # vbar = tqdm(vbar, total=len(self.valid_loader))
                    vbar = tqdm(vbar, desc=('%20s' * 3) % ('Phase' ,'Epoch', 'Total Loss'), total=len(self.valid_loader))
                    for step, batch_data in vbar:
                        self.val_loss, predictions, labels = self.val_step(batch_data)
                        
                        validation_predictions = torch.cat([validation_predictions, predictions], dim=0)
                        validation_labels = torch.cat([validation_labels, labels], dim=0)
                        
                        if self.epoch != 0: self.val_losses.append(self.val_loss)
                        pf = '%20s' * 3 # print format
                    print(pf % ("Validation", f'{self.epoch}/{self.epochs}', self.val_loss.item()))   
                    del vbar
                    # print(len(validation_predictions), len(validation_predictions[0]), len(validation_labels))


                    # PLot Losses
                    if self.epoch != 0: self.plot_loss()
                    
                    if self.val_loss < self.best_loss:
                        self.best_loss=self.val_loss

                    acc1, acc5 = accuracy(train_predictions, train_labels, topk=(1, 5))
                    print('Training Acc@1 {} Acc@5 {}'.format(acc1, acc5))
                    acc1, acc5 = accuracy(validation_predictions, validation_labels, topk=(1, 5))
                    print('Validation Acc@1 {} Acc@5 {}'.format(acc1, acc5))
                    
                    
                    del train_predictions, train_labels
                    del validation_predictions, validation_labels

                except Exception as _:
                    print('ERROR in training steps.')
                    raise
                # try:
                #     # save(conf=self.conf, save_dir=self.save_dir, model_name=self.model_name, model=self.linear_classifier, epoch=self.epoch, val_loss=self.val_loss, best_loss=self.best_loss, optimizer=self.optimizer)
                # except Exception as _:
                #     print('ERROR in evaluate and save model.')
                #     raise
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
    # -------------------------------------------------------save Model-------------------------------------------
    def save(self):
        # create config object
        conf = json.dumps(self.conf)
        # f = open(self.save_dir + "/config.json","w")
        # f.write(conf)
        # f.close()
        # save model
        save_ckpt_dir = osp.join(self.save_dir, 'weights')
        if not osp.exists(save_ckpt_dir):
            os.makedirs(save_ckpt_dir)
        filename = osp.join(save_ckpt_dir, self.model_name + "-" + str(self.epoch) + '.pt')
        torch.save(self.model.state_dict(), filename)
        if self.val_loss == self.best_loss:
            best_filename = osp.join(save_ckpt_dir, 'best_{}.pt'.format(self.model_name, self.epoch))
            if osp.exists(best_filename):
                os.remove(best_filename)
            shutil.copyfile(filename, best_filename)

Linear_Validator().run()