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
from models.layers import ModelBase
from models.layers import MLP
import copy



class SimCLR_MODEL(nn.Module):
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

    def contrastive_loss(x, t=0.5):
        x = F.normalize(x, dim=1)
        x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
        x_scale = x_scores / t   # scale with temperature

        # (2N-1)-way softmax without the score of i-th entry itself.
        # Set the diagonals to be large negative values, which become zeros after softmax.
        x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

        # targets 2N elements.
        targets = torch.arange(x.size()[0])
        targets[::2] += 1  # target of 2k element is 2k+1
        targets[1::2] -= 1  # target of 2k+1 element is 2k
        return F.cross_entropy(x_scale, targets.long().to(x_scale.device))

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

        p1 = self.predictor(z1) # NxC
        p2 = self.projector(z) # NxC

        loss = self.contrastive_loss(p)

        return (z1, p1), loss, [loss, loss, loss]