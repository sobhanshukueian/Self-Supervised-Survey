import torch.nn as nn
import torch
import torchvision.utils
import torchvision

from configs import model_config
from utils import off_diagonal

class BarlowTwins(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet50(num_classes=model_config["HIDDEN_SIZE"], zero_init_residual=True)
        self.backbone.fc = nn.Sequential(
            self.backbone.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(model_config["HIDDEN_SIZE"], model_config["EMBEDDING_SIZE"])
        )

        sizes = [model_config["EMBEDDING_SIZE"], model_config["HIDDEN_SIZE"], model_config["HIDDEN_SIZE"], model_config["HIDDEN_SIZE"], model_config["EMBEDDING_SIZE"]]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2=None):
        if y2 == None:
            return self.backbone(y1)
            
        r1 = self.backbone(y1)
        r2 = self.backbone(y2)

        z1 = self.bn(self.projector(r1))
        z2 = self.bn(self.projector(r2))
        return z1, z2


def barlow_twins_loss(z1, z2, lambda_reg=5e-3):
    """
    # Computes the Barlow Twins loss given two batches of encoded feature vectors.

    # Args:
    #     z1: A batch of encoded feature vectors of shape (batch_size, feature_dim).
    #     z2: A batch of encoded feature vectors of shape (batch_size, feature_dim).
    #     lambda_reg: A float specifying the regularization strength.

    # Returns:
    #     The Barlow Twins loss value as a scalar tensor.
    # """
    # # Normalize the feature vectors
    # z1_norm = z1 / z1.norm(dim=1, keepdim=True)
    # z2_norm = z2 / z2.norm(dim=1, keepdim=True)

    # # Compute the cross-correlation matrix
    # c = torch.mm(z1_norm.T, z2_norm)

    # # Compute the loss
    # eye = torch.eye(z1.size(1), device=z1.device)
    # loss = (1 / z1.size(1)) * torch.sum((c - eye) ** 2) + lambda_reg * torch.sum(c ** 2)
        # empirical cross-correlation matrix
    
    c = z1.T @ z2

    # sum the cross-correlation matrix between all gpus
    # c.div_(BATCH_SIZE)
    # torch.distributed.all_reduce(c)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + 0.0051 * off_diag
    return loss

# temp1 = torch.rand((6, 3, 32, 32))
# temp2 = torch.rand((6, 3, 32, 32))
# temp_model = BarlowTwins()
# res = temp_model(temp1, temp2)
# print(res[0].size(), res[1].size())