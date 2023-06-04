import torch.nn as nn
import torch
import torchvision.utils
import torchvision
import torch.nn.functional as F
import torch.nn.init as init

import copy
from utils import EMA, adjust_loss_weights
from configs import model_config

#create the Siamese Neural Network
class MOCO(nn.Module):

    def __init__(self, in_features=512, hidden_size=4096, embedding_size=256, projection_size=256, projection_hidden_size=2048, batch_norm_mlp=True):
        super(MOCO, self).__init__()
        self.online = self.get_representation()

        self.online.mean = nn.Linear(model_config["EMBEDDING_SIZE"], model_config["EMBEDDING_SIZE"])
        self.online.var = nn.Linear(model_config["EMBEDDING_SIZE"], model_config["EMBEDDING_SIZE"])
        self.predictor = self.get_linear_block()

        init.zeros_(self.online.var.weight)
        init.zeros_(self.online.mean.weight)
        # init.zeros_(self.predictor.weight)
        

        self.target = self.get_representation()
        self.target.mean = nn.Linear(model_config["EMBEDDING_SIZE"], model_config["EMBEDDING_SIZE"])
        self.target.var = nn.Linear(model_config["EMBEDDING_SIZE"], model_config["EMBEDDING_SIZE"])

        for param_t, param_o in zip(self.target.parameters(), self.online.parameters()):
            param_t.data.copy_(param_o.data)  # initialize
            param_t.requires_grad = False  # not update by gradient

        self.target = self.get_target()
        self.ema = 0.999

        self.LeakyReLU = nn.LeakyReLU(0.2)
    
    @torch.no_grad()
    def get_target(self):
        return copy.deepcopy(self.online)

    def get_linear_block(self):
        return nn.Sequential(
            nn.Linear(model_config["EMBEDDING_SIZE"], model_config["HIDDEN_SIZE"]),
            nn.BatchNorm1d(model_config["HIDDEN_SIZE"]),
            nn.ReLU(inplace=True),
            nn.Linear(model_config["HIDDEN_SIZE"], model_config["EMBEDDING_SIZE"])
        )

    def get_representation(self):
        backbone = torchvision.models.resnet50(num_classes=model_config["EMBEDDING_SIZE"])
        proj = self.get_linear_block()
        return nn.Sequential(backbone, proj)

    @torch.no_grad()
    def update_moving_average(self):
        for param_t, param_o in zip(self.target.parameters(), self.online.parameters()):
            param_t.data = param_t.data * self.ema + param_o.data * (1. - self.ema)
            
    def reparameterization(self, mean, logvar):
        var = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(var)      # sampling epsilon        
        z = mean + var * epsilon                          # reparameterization trick
        return z

    def byol_loss(self, x, y):
        # L2 normalization
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        loss = 2 - 2 * (x * y).sum(dim=-1)
        return loss

    def iso_kl(self, mean, log_var):
        # indices = find_inf_nan_indices(kl)
        # print(indices)
        # print(log_var)
        # if torch.isnan(kl) or torch.isinf(kl):
        # print("log_var: ", log_var)
        # print("log_var.exp: ", log_var.exp())
        # print("mean.pow: ", mean.pow(2))
        return - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    def kl_divergence(self, mu1, log_var1, mu2, log_var2):
        var1 = torch.exp(log_var1)
        var2 = torch.exp(log_var2)

        term1 = (var1 / var2 - 1).sum(dim=1)
        term2 = ((mu2 - mu1).pow(2) / var2).sum(dim=1)
        term3 = (log_var2 - log_var1).sum(dim=1)
        kl_div = 0.5 * (term1 + term2 + term3)
        
        return kl_div.sum()

    def cosine_sim(self, tensor1, tensor2):
        return torch.abs(F.cosine_similarity(tensor1, tensor2, dim=1)).mean()

    def forward_once(self, x):
        embedding_o = self.online(x)
        z_o_p = self.predictor(embedding_o)

        mean_o = self.online.mean(self.LeakyReLU(embedding_o))
        logvar_o = self.online.var(self.LeakyReLU(embedding_o))
        # z_o = self.reparameterization(mean_o, logvar_o)

        with torch.no_grad():
            embedding_tar = self.target(x).detach()
            mean_tar = self.target.mean(self.LeakyReLU(embedding_tar)).detach()
            logvar_tar = self.target.var(self.LeakyReLU(embedding_tar)).detach()
            # z_tar = self.reparameterization(mean_tar, logvar_tar).detach()

        distance_loss = self.byol_loss(z_o_p, embedding_tar).mean()
        # distance_loss = F.pairwise_distance(z_o_p, z_tar, keepdim = True).squeeze().mean()

        kl_loss = self.kl_divergence(mean_o, logvar_o, mean_tar, logvar_tar)

        iso_kl_loss = self.iso_kl(mean_o, logvar_o)
        iso_kl_loss += self.iso_kl(mean_tar, logvar_tar)

        if torch.isnan(distance_loss) or torch.isinf(distance_loss):
            print("------------------------")
            # print("z_o: ", z_o)
            print("z_o_p: ", z_o_p)
            # print("z_tar: ", z_tar)
            print("logvar_o: ", logvar_o)
            print("logvar_tar: ", logvar_tar)


        return kl_loss, distance_loss, iso_kl_loss, embedding_o
    
    def init_var(self):
        init.zeros_(self.online.var.weight)
        init.zeros_(self.online.mean.weight)

    def my_activation(self, x):
        return 1 - torch.pow(1 + torch.pow(x, 3) + 0.001, -1)


    def forward(self, x1, x2=None, weight=0):
        if x2 is None:
            return self.online(x1)
        
        with torch.no_grad():  # no gradient to keys
            self.update_moving_average()

        kl_loss1, distance_loss1, iso_kl_loss1, embedding_o1 = self.forward_once(x1)
        kl_loss2, distance_loss2, iso_kl_loss2, embedding_o2 = self.forward_once(x2)


        kl_total = kl_loss1 + kl_loss2
        iso_kl_total = iso_kl_loss1 + iso_kl_loss2
        distance_total = distance_loss1 + distance_loss2
        # if weight > 0:
        # kl_total *= 0.001
        iso_kl_total *= 0.00001
        # distance_total *= 

        losses = [torch.clone(kl_total), torch.clone(iso_kl_total), torch.clone(distance_total)]
        
        # iso_kl_total = self.my_activation(torch.abs(iso_kl_total))
        # kl_total = self.my_activation(torch.abs(kl_total))    
        # distance_total = self.my_activation(torch.abs(distance_total))

        total_loss = iso_kl_total + distance_total

        # print("kl_loss: ", kl_total)
        # print("distance_total: ", distance_total)
        # print("iso_kl_total: ", iso_kl_total)

        return(embedding_o1, embedding_o2), total_loss, losses
# temp1 = torch.rand((6, 3, 32, 32))
# temp2 = torch.rand((6, 3, 32, 32))
# temp_model = MOCO()
# ress = temp_model(temp1, temp2)[0]
# for res in ress:
#     print(res[0].size(), res[1].size(), res[2].size(), res[3].size())