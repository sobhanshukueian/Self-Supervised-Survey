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
from models.variation import GaussianProjection
import copy



class SimCLR_VAR_MODEL(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, dim=model_config["EMBEDDING_SIZE"], arch='resnet18'):
        """
        dim: feature dimension (default: 2048)
        arch: Backbone architecture
        """
        super(SimCLR_VAR_MODEL, self).__init__()

        # create the encoder
        self.encoder = ModelBase(feature_dim=dim, arch=arch)
        self.encoder_gaussian = GaussianProjection()

        self.projector = MLP(dim, dim)

    def contrastive_loss(self, p1, p2):
        mean = torch.cat([p1, p2], dim=0)
        cos_sim = F.cosine_similarity(mean[:,None,:], mean[None,:,:], dim=-1)
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        cos_sim = cos_sim #/ self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        loss = nll.mean()

        return loss

    def iso_kl(self, mean, log_var):
        return - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

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

        p1 = self.projector(z1) # NxC
        p2 = self.projector(z2) # NxC

        p1_mean, p1_var = self.encoder_gaussian(p1)
        p2_mean, p2_var = self.encoder_gaussian(p2)
        
        
        iso_kl_total = 0.001*(self.iso_kl(p1_mean, p1_var) + self.iso_kl(p1_mean, p1_var))

        gaussian_loss = self.contrastive_loss(p1_mean, p2_mean)

        contrastive_loss = self.contrastive_loss(p1, p2)

        loss = contrastive_loss + gaussian_loss + iso_kl_total
        return (z1, p1), loss, [contrastive_loss, gaussian_loss, iso_kl_total]