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
from models.layers import MLP
import copy
from models.layers import ModelBase




class SimSiam_MODEL(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, dim=model_config["EMBEDDING_SIZE"], arch='resnet18'):
        """
        dim: feature dimension (default: 2048)
        arch: Backbone architecture
        """
        super(SimSiam_MODEL, self).__init__()

        # create the encoder
        self.backbone = ModelBase(feature_dim=dim, arch=arch)

        self.projector = MLP(dim, dim)

        '''self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )'''

        self.encoder = self.backbone

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, model_config["HIDDEN_SIZE"], bias=False),
                                        nn.BatchNorm1d(model_config["HIDDEN_SIZE"]),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(model_config["HIDDEN_SIZE"], dim)) # output layer

        self.criterion = nn.CosineSimilarity(dim=1).to(model_config["device"])

    def forward(self, x1, x2, train=False):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC
        #-------------------------------------
        h1 = self.projector(z1) # NxC
        h2 = self.projector(z2) # NxC
        #-------------------------------------
        p1 = self.predictor(h1) # NxC
        p2 = self.predictor(h2) # NxC

        loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5

        return (z1, p1), loss, [loss, loss, loss]