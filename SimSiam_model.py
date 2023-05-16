from torch import nn
from configs import model_config
import torch
import torchvision

class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        hidden_dim = out_dim
        self.num_layers = num_layers

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)  # Page:5, Paragraph:2
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, out_dim=256):
        super().__init__()
        hidden_dim = int(out_dim / 4)

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x


class SimSiam(nn.Module):
    def __init__(self):
        super(SimSiam, self).__init__()
        self.backbone = torchvision.models.resnet50(num_classes=model_config["EMBEDDING_SIZE"])  # Output of last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.backbone.fc = nn.Sequential(
            self.backbone.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(model_config["EMBEDDING_SIZE"], model_config["HIDDEN_SIZE"])
        )
        self.projector = projection_MLP(model_config["HIDDEN_SIZE"], model_config["EMBEDDING_SIZE"], 2)

        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

        self.predictor = prediction_MLP(model_config["EMBEDDING_SIZE"], model_config["EMBEDDING_SIZE"])

    def forward(self, im_aug1, im_aug2):

        z1 = self.encoder(im_aug1)
        z2 = self.encoder(im_aug2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return {'z1': z1, 'z2': z2, 'p1': p1, 'p2': p2}
    def forward_single(self, im_aug1):

        z1 = self.encoder(im_aug1)

        p1 = self.predictor(z1)

        return z1, p1


from torch import nn


class SimSiamLoss(nn.Module):
    def __init__(self, version='simplified'):
        super().__init__()
        self.ver = version

    def asymmetric_loss(self, p, z):
        if self.ver == 'original':
            z = z.detach()  # stop gradient

            p = nn.functional.normalize(p, dim=1)
            z = nn.functional.normalize(z, dim=1)

            return -(p * z).sum(dim=1).mean()

        elif self.ver == 'simplified':
            z = z.detach()  # stop gradient
            return - nn.functional.cosine_similarity(p, z, dim=-1).mean()

    def forward(self, z1, z2, p1, p2):

        loss1 = self.asymmetric_loss(p1, z2)
        loss2 = self.asymmetric_loss(p2, z1)

        return 0.5 * loss1 + 0.5 * loss2

    def forward_simgle(self, p1, z2):

        loss1 = self.asymmetric_loss(p1, z2)

        return 0.5 * loss1



# temp1 = torch.rand((6, 3, 32, 32))
# temp2 = torch.rand((6, 3, 32, 32))
# temp_model = SimSiam()
# res = temp_model(temp1, temp2)
# print(res["z1"].size(), res["z2"].size(), res["p1"].size(), res["p2"].size())