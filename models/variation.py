#create the Siamese Neural Network
import math

import torch.nn as nn
import torch
import torchvision.utils
import torchvision
import torch.nn.functional as F

from configs import model_config
import torch.nn.init as init

class GaussianProjection(nn.Module):

    def __init__(self):
        super(GaussianProjection, self).__init__()
        # self.projection = self.get_mlp_block(model_config["EMBEDDING_SIZE"])
        self.mean = nn.Linear(model_config["PROJECTION_SIZE"], model_config["PROJECTION_SIZE"])
        self.var = nn.Linear(model_config["PROJECTION_SIZE"], model_config["PROJECTION_SIZE"])

        init.zeros_(self.var.weight)
        init.zeros_(self.mean.weight)


    def get_mlp_block(self, in_ch):
        return nn.Sequential(
            nn.Linear(in_ch, model_config["HIDDEN_SIZE"]),
            nn.BatchNorm1d(model_config["HIDDEN_SIZE"]),
            nn.ReLU(inplace=True),
            nn.Linear(model_config["HIDDEN_SIZE"], model_config["HIDDEN_SIZE"]),
            nn.BatchNorm1d(model_config["HIDDEN_SIZE"]),
            nn.ReLU(inplace=True),
            nn.Linear(model_config["HIDDEN_SIZE"], model_config["PROJECTION_SIZE"])
        )

    def forward(self, x1):
        # x1 = self.projection(x1)
        x1_mean = self.mean(x1)
        x1_var = self.var(x1)

        return x1_mean, x1_var