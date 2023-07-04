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
from models.variation import GaussianProjection
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




class SimSiam_VAR_MODEL(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, dim=model_config["EMBEDDING_SIZE"], arch='resnet18'):
        """
        dim: feature dimension (default: 2048)
        arch: Backbone architecture
        """
        super(SimSiam_VAR_MODEL, self).__init__()

        # create the encoder
        self.backbone = ModelBase(feature_dim=dim, arch=arch)

        self.projector = MLP(dim, dim)

        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

        self.encoder_gaussian = GaussianProjection()

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, model_config["HIDDEN_SIZE"], bias=False),
                                        nn.BatchNorm1d(model_config["HIDDEN_SIZE"]),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(model_config["HIDDEN_SIZE"], dim)) # output layer

        self.criterion = nn.CosineSimilarity(dim=1).to(model_config["device"])

    def iso_kl(self, mean, log_var):
        return - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())


    def gaussian_mean_loss(self, mean1, mean2):
        mean = torch.cat([mean1, mean2], dim=0)
        cos_sim = F.cosine_similarity(mean[:,None,:], mean[None,:,:], dim=-1)
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        cos_sim = cos_sim #/ self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        loss_gaussian = nll.mean()
        return loss_gaussian

    
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

        p1_mean, p1_var = self.encoder_gaussian(p1)
        p2_mean, p2_var = self.encoder_gaussian(p2)

        z1_mean, z1_var = self.encoder_gaussian(z1)
        z2_mean, z2_var = self.encoder_gaussian(z2)

        iso_kl_total = 0.001 * self.iso_kl(p1_mean, p1_var) +  0.001 * self.iso_kl(p2_mean, p2_var)

        loss_gaussian_total = self.gaussian_mean_loss(p1_mean, z2_mean.detach()) + self.gaussian_mean_loss(p2_mean, z1_mean.detach())

        contrastive_loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5

        loss = contrastive_loss + loss_gaussian_total + iso_kl_total

        return (z1, p1), loss, [contrastive_loss, loss_gaussian_total, iso_kl_total]