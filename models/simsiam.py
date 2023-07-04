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


class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, feature_dim=128, arch=None):
        super(ModelBase, self).__init__()

        # use split batchnorm
        norm_layer = nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        # note: not normalized here
        return x




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

        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

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

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5

        return (z1, p1), loss, [loss, loss, loss]