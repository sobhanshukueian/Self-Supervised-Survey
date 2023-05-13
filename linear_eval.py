
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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


from cifar_dataset import train_dataloader, train_val_dataloader, test_dataloader, vis_dataloader
from vis import show_batch
from configs import model_config
from utils import LARS, off_diagonal, get_color, get_colors, get_params_groups, LinearClassifier
from BYOL_model import BYOLNetwork
from Barlow_model import BarlowTwins
from main_utils import get_optimizer, get_model, eval_knn


MODE = "barlow" #@param
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



class Linear_Validator:
    # -----------------------------------------------INITIALIZE TRAINING-------------------------------------------------------------
    def __init__(self, device=device, epochs=EPOCHS, batch_size=BATCH_SIZE, save_dir=SAVE_DIR, train_loader=train_dataloader, valid_loader=test_dataloader, weights=WEIGHTS, verbose=VERBOSE, visualize_plots=VISUALIZE_PLOTS, save_plots=SAVE_PLOTS, model_name=MODEL_NAME, mode=MODE):
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
        self.model, self.linear_classifier = self.get_model()
        if self.verbose > 2:
            self.count_parameters()

        # Get optimizer
        self.optimizer = self.get_optimizer()
    
        # tensorboard
        # self.tblogger = SummaryWriter(self.save_dir) 

# ----------------------------------------------------INITIALIZERS-------------------------------------------------------------------------
    # Get Model 
    def get_model(self):
        if self.mode == 'byol':
            model = BYOLNetwork().to(self.device)
            if self.weights:  
                print(f'Loading state_dict from {self.weights} for fine-tuning...')
                model.load_state_dict(torch.load(self.weights))
        if self.mode == 'barlow':
            model = BarlowTwins().to(self.device)
            if self.weights:  
                print(f'Loading state_dict from {self.weights} for fine-tuning...')
                model.load_state_dict(torch.load(self.weights))
        elif self.mode == 'supervised':
            model = torchvision.models.resnet18(pretrained=True)
            model.fc = nn.Sequential(
                nn.Linear(model_config["HIDDEN_SIZE"], model_config["EMBEDDING_SIZE"])
                )
            model = model.to(device)

        model.eval()
        linear_classifier = LinearClassifier(model_config["EMBEDDING_SIZE"])
        linear_classifier = linear_classifier.cuda()
        return model, linear_classifier

    def get_optimizer(self, optimizer="SGD", lr0=0.0008, momentum=0.9):
        assert optimizer == 'SGD' or 'Adam' or 'LARS', 'ERROR: unknown optimizer, use SGD defaulted'
        if optimizer == 'SGD':
            optim = torch.optim.SGD(self.linear_classifier.parameters(), lr=lr0, momentum=momentum, nesterov=True)
        elif optimizer == 'Adam':
            optim = torch.optim.Adam(self.linear_classifier.parameters(), lr=lr0, weight_decay=1e-6)

        if self.verbose > 1:
            print(f"{'optimizer:'} {type(optim).__name__}")
        self.conf['Optimizer'] = f"{'optimizer:'} {type(optim).__name__}"
        return optim

    # Loss Function Definition
    def compute_loss(self, predictions, labels):
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(predictions, labels)
        return loss
    
    def count_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        self.conf["Parameter_size"] = total_params
# -------------------"------------------------------------------------------------TRAINING PROCESS-----------------------------------------------
    @staticmethod
    def prepro_data(batch_data, device):
        inputs, _, labels = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device)
        labels = labels.view(-1)
        return inputs.to(device), labels.to(device)

    # Each Train Step
    def train_step(self, batch_data):
        inputs, labels = self.prepro_data(batch_data, self.device)
        with amp.autocast(enabled=self.device != 'cpu'):
            outputs = self.model(inputs)
            # print(outputs.size())
            outputs = self.linear_classifier(outputs)
            # print(labels.size())

            # outputs = torch.max(outputs, dim=1)[1]
            # print(outputs.size())


            loss = self.compute_loss(outputs, labels)

        # state_dict_before = copy.deepcopy(self.model.state_dict())

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.cpu().detach().numpy(), outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()


    # Each Validation Step
    def val_step(self, batch_data):
        self.model.eval()
        inputs, labels = self.prepro_data(batch_data, self.device)
        outputs = self.model(inputs)
        outputs = self.linear_classifier(outputs)
        loss = self.compute_loss(outputs, labels)

        return loss.cpu().detach().numpy(), outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()

    # Training Process
    def run(self):
        try:
            # training process prerequisite
            self.start_time = time.time()
            print('Start Training Process \nTime: {}'.format(time.ctime(self.start_time)))
            self.scaler = amp.GradScaler(enabled=self.device != 'cpu')
            self.best_loss = np.inf

            # Epoch Loop
            for self.epoch in range(0, self.epochs):
                try:
                    self.conf["Trained_epoch"] = self.epoch

                    train_predictions = []
                    train_labels = []

                    validation_predictions = []
                    validation_labels = []

                    # ############################################################Train Loop
                    # if self.epoch != 0:
                    # Training loop
                    self.model.train(True)
                    pbar = enumerate(self.train_loader)
                    pbar = tqdm(pbar, desc=('%20s' * 3) % ('Phase' ,'Epoch', 'Total Loss'), total=self.max_stepnum)
                    for step, batch_data in pbar:
                        self.train_loss, predictions, labels = self.train_step(batch_data)
                        train_predictions.extend(predictions)
                        train_labels.extend(labels)
                        self.train_losses.append(self.train_loss)
                        pf = '%20s' * 3 # print format
                    print(pf % ("Train", f'{self.epoch}/{self.epochs}', self.train_loss.item()))                 
                    del pbar



                    # ############################################################Validation Loop

                    # Validation Loop
                    vbar = enumerate(self.valid_loader)
                    # vbar = tqdm(vbar, total=len(self.valid_loader))
                    vbar = tqdm(vbar, desc=('%20s' * 3) % ('Phase' ,'Epoch', 'Total Loss'), total=len(self.valid_loader))
                    for step, batch_data in vbar:
                        self.val_loss, predictions, labels = self.val_step(batch_data)
                        validation_predictions.extend(predictions)
                        validation_labels.extend(labels)
                        if self.epoch != 0: self.val_losses.append(self.val_loss)
                        # vbar.set_description(f"Epoch: {self.epoch}/{self.epochs}\tValidation Loss: {self.val_loss}  ")
                        pf = '%20s' * 3 # print format
                    print(pf % ("Train Validation", f'{self.epoch}/{self.epochs}', self.val_loss.item()))   
                    del vbar
                    # print(len(validation_predictions), len(validation_predictions[0]), len(validation_labels))


                    # PLot Losses
                    if self.epoch != 0: self.plot_loss()
                    
                    if self.val_loss < self.best_loss:
                        self.best_loss=self.val_loss

                    train_acc = self.compute_acc(train_predictions, train_labels)
                    validation_acc = self.compute_acc(validation_predictions, validation_labels)
                    print("Train Accuracy: {}\nValidation Accuracy: {}".format(train_acc, validation_acc))

            
                except Exception as _:
                    print('ERROR in training steps.')
                    raise
                try:
                    self.save()
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
    def compute_acc(self, predicted, labels):
        # print(predicted.size(), labels.size())
        predicted = np.argmax(predicted, 1)  
        correct = (predicted == labels).sum().item() 
        total = len(labels)
        return (100 * correct / total)


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