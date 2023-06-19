# MOCO without projection layer 

import torch.nn as nn
import torch
import torchvision.utils
import torchvision
import torch.nn.functional as F
from torchvision.models import resnet
import torchvision.models as torchvision_models
from functools import partial
import torch.nn.init as init


from configs import model_config
import copy


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class Barlow_model(nn.Module):
    def __init__(self):
        super(Barlow_model, self).__init__()

        # create the encoders
        self.backbone = partial(torchvision_models.__dict__["resnet50"], zero_init_residual=True)(num_classes=model_config["PROJECTION_SIZE"])
        self.backbone.fc = nn.Identity()
        projector = '8192-8192-8192'
        sizes = [2048] + list(map(int, projector.split('-')))
        # print(sizes)
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(model_config["batch_size"])

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + 0.0051 * off_diag
        return (z1, z2), loss, (loss, loss, loss)


temp_model = Barlow_model()
temp_data = torch.rand((10, 3, 32, 32))
temp_data2 = torch.rand((10, 3, 32, 32))
result = temp_model(temp_data, temp_data2)

# print(result)