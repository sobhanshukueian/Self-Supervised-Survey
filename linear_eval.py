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
from torchvision.datasets import CIFAR10
import matplotlib
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from utils.utils import CosineAnnealingWarmupRestarts, set_logging
import random

import torch
import torchvision.utils
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.cuda import amp

from data.stl_dataset import get_stl_data
from data.cifar_dataset import get_cifar_test
from utils.vis import plot_embeddings
from configs import model_config
from utils.utils import get_color, get_colors, count_parameters, save, adjust_learning_rate, get_params_groups, LinearClassifier, accuracy
from utils.main_utils import get_optimizer, get_model
from knn_eval import knn_monitor



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def reproducibility(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

reproducibility(3)

g = torch.Generator()
g.manual_seed(3)

if model_config["dataset"] == "STL10":
    train_dataloader, test_dataloader = get_stl_data(seed_worker, g)
elif model_config["dataset"] == "CIFAR10":
    train_dataloader, test_dataloader = get_cifar_test(seed_worker, g)
else:
    train_dataloader, test_dataloader = get_cifar_test(seed_worker, g)
    train_dataloader = test_dataloader


class Linear_Validator:
    # -----------------------------------------------INITIALIZE TRAINING-------------------------------------------------------------
    def __init__(self, train_loader=train_dataloader, valid_loader=test_dataloader):
        self.device = model_config['device']
        self.save_dir = model_config['SAVE_DIR']
        self.batch_size = model_config['batch_size']
        self.epochs = model_config['EPOCHS']
        self.model_name = model_config['MODEL_NAME']
        self.weights = model_config['WEIGHTS']
        self.visualize_plots = model_config["VISUALIZE_PLOTS"]
        self.save_plots = model_config["SAVE_PLOTS"]
        # 0 == nothing || 1 == model architecture || 2 == print optimizer || 3 == model parameters
        self.train_losses=[]
        self.val_losses=[]
        self.val_losses_s=[]
        self.conf = {'Basic configs': model_config, 'Max_iter_num' : '', 'Epochs' : self.epochs, 'Trained_epoch' : 0, 'Optimizer' : '', 'Parameter_size' : '', "Model" : ''}
        self.ckpt = False
        self.resume = model_config['RESUME']
        self.resume_dir = model_config["RESUME_DIR"]
        self.start_epoch = 0

        
        # get data loader

        self.logger = set_logging(self.save_dir, self.model_name)

        # get data loader
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.max_stepnum = len(self.train_loader)
        self.conf["Max_iter_num"] = self.max_stepnum

        
        # get model 
        self.model, self.conf, self.ckpt = get_model(model_config["MODEL_NAME"], self.conf, self.resume, self.resume_dir, self.weights)
        self.model = self.model.encoder_q.to(self.device)
        self.model = self.model.eval()

        self.linear_classifier = LinearClassifier(model_config["EMBEDDING_SIZE"])
        self.linear_classifier = self.linear_classifier.to(self.device)

        #--------------------------------

        self.logger = set_logging(self.save_dir, self.model_name)

        # with torch.no_grad():
        #   self.model.fc.weight.data.normal_(mean= 0.0 , std= 0.01)
        #   self.model.fc.bias.data.zero_()
        #-------------------------------------------------

        self.conf = count_parameters(self.logger, self.model, self.conf)

        self.optimizer, self.conf = get_optimizer(self.logger, get_params_groups(self.linear_classifier), self.conf, self.resume, self.ckpt, model_config['OPTIMIZER'], lr0=model_config["LEARNING_RATE"], momentum=model_config["MOMENTUM"], weight_decay=model_config["WEIGHT_DECAY"])
        # self.optimizer = torch.optim.SGD(get_params_groups(self.model), lr=0.06, weight_decay=5e-4, momentum=0.9)

        if self.resume:
            self.start_epoch = self.ckpt["epoch"] + 1
            self.conf['resume'] += f" from epoch {self.start_epoch}"

        self.criterion = torch.nn.CrossEntropyLoss()
        
    
# -------------------"------------------------------------------------------------TRAINING PROCESS-----------------------------------------------

    def train_step(self):
        self.model.train(False)
        self.linear_classifier.train(True)

        train_predictions = torch.tensor([])
        train_labels = torch.tensor([])
        lr = adjust_learning_rate(self.optimizer, self.epoch, model_config["LEARNING_RATE"])

        pbar = enumerate(self.train_loader)
        pbar = tqdm(pbar, desc=('%20s' * 6) % ('Phase' ,'Epoch', 'Total Loss', 'Learning Rate', 'Acc@1', 'Acc@5'), total=self.max_stepnum)                        
        self.logger.warning(('%20s' * 6) % ('Phase' ,'Epoch', 'Total Loss', 'Learning Rate', 'Acc@1', 'Acc@5'))
        total_loss, total_num = 0.0, 0
        for step, (images, targets) in pbar:
            images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True).view(-1)

            with torch.no_grad():
                outputs = self.model(images)

            outputs = self.linear_classifier(outputs)

            train_loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            
            total_num += self.train_loader.batch_size
            total_loss += train_loss.item() * self.train_loader.batch_size

            train_predictions = torch.cat([train_predictions, outputs.cpu().detach()], dim=0)
            train_labels = torch.cat([train_labels, targets.cpu().detach()], dim=0)

            if self.epoch != 0: 
                self.train_losses.append(total_loss/total_num)

            pbar.set_postfix({'loss':total_loss/total_num})

            # del train_predictions
            # del train_labels


        acc1, acc5 = accuracy(train_predictions, train_labels, topk=(1, 5))
        print('%20s' * 6  % ("Train", f'{self.epoch}/{self.epochs}', total_loss/total_num, lr, acc1, acc5))     
        self.logger.warning('%20s' * 6  % ("Train", f'{self.epoch}/{self.epochs}', total_loss/total_num, lr, acc1, acc5))
        return acc1, acc5

    def validation(self):
        self.linear_classifier.eval()

        # val_labels = []
        # val_embeddings = []

        validation_predictions = torch.tensor([])
        validation_labels = torch.tensor([])


        # Validation Loop
        vbar = enumerate(self.valid_loader)
        vbar = tqdm(vbar, desc=('%20s' * 5) % ('Phase' ,'Epoch', 'Total Loss', 'Acc@1', 'Acc@5'), total=len(self.valid_loader))
        self.logger.warning(('%20s' * 5) % ('Phase' ,'Epoch', 'Total Loss', 'Acc@1', 'Acc@5'))

        for step, (images, targets) in vbar:
            images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True).view(-1)

            outputs = self.model(images)
            outputs = self.linear_classifier(outputs)
            val_loss = self.criterion(outputs, targets)

            # if self.epoch != 0: 
            #     self.val_losses.append(val_loss)
            
            # val_embeddings.extend(val_embeds[0])
            # val_labels.extend(val_targets)
        validation_predictions = torch.cat([validation_predictions, outputs.cpu().detach()], dim=0)
        validation_labels = torch.cat([validation_labels, targets.cpu().detach()], dim=0)

        acc1, acc5 = accuracy(validation_predictions, validation_labels, topk=(1, 5))

        print('%20s' * 5 % ("Validation", f'{self.epoch}/{self.epochs}', val_loss, acc1, acc5))
        self.logger.warning('%20s' * 5 % ("Validation", f'{self.epoch}/{self.epochs}', val_loss, acc1, acc5))
        return acc1, acc5
        # del validation_predictions
        # del validation_labels
        
        # PLot Losses
        # if self.epoch != 0: self.plot_loss()
        # PLot Embeddings
        # plot_embeddings(self.epoch, np.array(val_embeddings), np.array(val_labels), 0)


    # Training Process
    def run(self):
        try:
            torch.cuda.empty_cache()

            # training process prerequisite
            self.start_time = time.time()
            self.conf["Time"] = time.ctime(self.start_time)
            print('Start Training Process \nTime: {}'.format(time.ctime(self.start_time)))
            self.logger.warning('Start Training Process \nTime: {}'.format(time.ctime(self.start_time)))
            print(torch.cuda.memory_allocated())
            
            filename = self.save_dir + "/Linear.csv"
            file_exists = os.path.isfile(filename)
            # Open the CSV file in append mode
            with open(filename, 'a', newline='') as file:
                writer = csv.writer(file)

                # If the file doesn't exist or doesn't contain the header row, add it
                if not os.path.isfile(filename) or 'epoch, train_acc@1, train_acc@5, validation_acc@1, validation_acc@5' not in open(filename).read():
                    writer.writerow(['epoch', 'train_acc@1', 'train_acc@5', 'validation_acc@1', 'validation_acc@5'])

            
            # Epoch Loop

            for self.epoch in range(self.start_epoch, self.epochs+1):
                try:
                    self.conf["Trained_epoch"] = self.epoch

                    train_acc1, train_acc5, valid_acc1, valid_acc5 = 0,0,0,0

                    validation_predictions = torch.tensor([])
                    validation_labels = torch.tensor([])

                    # ############################################################Train Loop
                    
                    if self.epoch != 0:
                        initial_params = [param.clone() for param in self.model.parameters()]
                        initial_classifier_params = [param.clone() for param in self.linear_classifier.parameters()]

                        train_acc1, train_acc5 = self.train_step()

                        self.sanity_check(self.model.parameters(), initial_params)
                        self.sanity_check(self.linear_classifier.parameters(), initial_classifier_params)

                        # del initial_classifier_params
                        # del initial_params

                    else : 
                        initial_params = [param.clone() for param in self.model.parameters()]
                        initial_classifier_params = [param.clone() for param in self.linear_classifier.parameters()]

                    # ############################################################Validation Loop

                    if self.epoch % model_config['VALIDATION_FREQ'] == 0 : 
                        print("--------------------------")
                        valid_acc1, valid_acc5 = self.validation()
                        print("--------------------------")

                    if self.epoch == 0:
                        self.sanity_check(self.model.parameters(), initial_params)
                        self.sanity_check(self.linear_classifier.parameters(), initial_classifier_params)

                        # del initial_classifier_params
                        # del initial_params

                    with open(filename, 'a', newline='') as file:
                        writer.writerow([self.epoch, train_acc1, train_acc5, valid_acc1, valid_acc5])

                        
                    save(conf=self.conf, save_dir=self.save_dir, model_name=self.model_name, model=self.linear_classifier, epoch=self.epoch, optimizer=self.optimizer)
         
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
    def sanity_check(self, parameters, initial_params):
        if any((param != initial_param).any() for param, initial_param in zip(parameters, initial_params)):
            print("=> Sanity checked : Failed. ⛔")
        else :
            print("=> Sanity check : Sucess 👌.")
            
    
    
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


if __name__ == '__main__':
    
    tmp_save_dir = model_config["SAVE_DIR"]
    temm = 0

    while os.path.exists(tmp_save_dir):
        temm += 1
        tmp_save_dir = f"{model_config['SAVE_DIR']}{temm}"

    model_config["SAVE_DIR"] = tmp_save_dir
    print("Save Project in {} directory.".format(model_config["SAVE_DIR"]))
    Linear_Validator().run()