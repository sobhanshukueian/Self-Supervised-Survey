import torch.nn as nn
import torch
import torchvision.utils
import torchvision
import torch.nn.functional as F

import copy
from utils import EMA

#create the Siamese Neural Network
class BYOLNetwork(nn.Module):

    def __init__(self, in_features=512, hidden_size=4096, embedding_size=256, projection_size=256, projection_hidden_size=2048, batch_norm_mlp=True):
        super(BYOLNetwork, self).__init__()
        self.online = self.get_rep_and_proj(in_features, embedding_size, hidden_size, batch_norm_mlp)
        self.predictor = self.get_cnn_block(projection_size, projection_size, projection_hidden_size)
        self.target = self.get_target()
        self.ema = EMA(0.99)
    
    @torch.no_grad()
    def get_target(self):
        return copy.deepcopy(self.online)

    def get_cnn_block(self, dim, embedding_size=256, hidden_size=2048, batch_norm_mlp=False):
        norm = nn.BatchNorm1d(hidden_size) #if batch_norm_mlp else nn.Identity()
        return nn.Sequential(
            nn.Linear(dim, hidden_size),
            norm,
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, embedding_size)
        )

    def get_rep_and_proj(self, in_features, embedding_size, hidden_size, batch_norm_mlp):
        self.backbone = torchvision.models.resnet50(num_classes=hidden_size)  # Output of last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.backbone.fc = nn.Sequential(
            self.backbone.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, in_features)
        )
        proj = self.get_cnn_block(in_features, embedding_size, hidden_size=hidden_size, batch_norm_mlp=batch_norm_mlp)
        return nn.Sequential(self.backbone, proj)

    @torch.no_grad()
    def update_moving_average(self):
        for online_params, target_params in zip(self.online.parameters(), self.target.parameters()):
            old_weight, up_weight = target_params.data, online_params.data
            target_params.data = self.ema.update_average(old_weight, up_weight)
    def byol_loss(self, x, y):
        # L2 normalization
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        loss = 2 - 2 * (x * y).sum(dim=-1)
        return loss

    def forward(self, x1, x2=None, return_embedding=False):
        if return_embedding or (x2 is None):
            return self.online(x1)

        # online projections: backbone + MLP projection
        x1_1 = self.online(x1)
        x1_2 = self.online(x2)

        # additional online's MLP head called predictor
        x1_1_pred = self.predictor(x1_1)
        x1_2_pred = self.predictor(x1_2)

        with torch.no_grad():
            # teacher processes the images and makes projections: backbone + MLP
            x2_1 = self.target(x1).detach_()
            x2_2 = self.target(x2).detach_()

        loss = (self.byol_loss(x1_1_pred, x2_1) + self.byol_loss(x1_2_pred, x2_2)).mean()

        return (x1_1_pred, x1_2_pred, x2_1, x2_2, x1_1), loss