import torch.nn as nn
import torch
import torchvision.utils
import torchvision
import torch.nn.functional as F

import copy
from utils import EMA
from configs import model_config

#create the Siamese Neural Network
class MOCO_Network(nn.Module):

    def __init__(self):
        super(MOCO_Network, self).__init__()
        
        self.online = self.get_representation()
        self.online.mean = self.get_linear_block()
        self.online.var = self.get_linear_block()
        self.online.predict = self.get_linear_block()


        self.target = self.get_target()
        # self.target.mean = self.copy_block(self.online.mean)
        # self.target.var = self.copy_block(self.online.var)
        # self.target.predict = self.copy_block(self.online.predict)

        self.ema = EMA(0.99)    
        self.f = nn.Flatten()

        self.loss = MyLoss( )

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
        return torchvision.models.resnet50(num_classes=model_config["EMBEDDING_SIZE"])

    @torch.no_grad()
    def update_moving_average(self):
        assert self.target is not None, 'target encoder has not been created yet'

        for online_params, target_params in zip(self.online.parameters(), self.target.parameters()):
            old_weight, up_weight = target_params.data, online_params.data
            target_params.data = self.ema.update_average(old_weight, up_weight)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)      # sampling epsilon        
        z = mean + var * epsilon                          # reparameterization trick
        return z

    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        embedding_o = self.online(x)
        mean_o = self.online.mean(embedding_o)
        var_o = self.online.var(embedding_o)
        z_o = self.reparameterization(mean_o, var_o)
        z_o_p = self.online.predict(z_o)
        # z_tar = self.target.predict(z_tar)

        with torch.no_grad():
            embedding_tar = self.target(x).detach()
            mean_tar = self.target.mean(embedding_tar).detach()
            var_tar = self.target.var(embedding_tar).detach()
            z_tar = self.reparameterization(mean_tar, var_tar).detach()

        return (embedding_o, mean_o, var_o, z_o, z_o_p), (embedding_tar, mean_tar, var_tar, z_tar)
    
    def forward(self, input1, input2):
        if input2 is None :
            return self.online(x)

        o1, t1 = self.forward_once(input1)
        o2, t2 = self.forward_once(input2)

        loss, losses = self.loss(o1, o2, t1, t2)

        return (o1[0], o1[-1]), loss, losses

# Define the Contrastive Loss Function
class MyLoss(torch.nn.Module):
    def __init__(self, margin=2):
        super(MyLoss, self).__init__()
        self.margin = margin

    def kl_divergence(self, mu1, log_var1, mu2, log_var2):
        max_logvar = 10.0
        log_var1 = torch.clamp(log_var1, max=max_logvar)
        log_var2 = torch.clamp(log_var2, max=max_logvar)

        
        var1 = torch.exp(log_var1)
        var2 = torch.exp(log_var2)

        term1 = (var1 / var2 - 1).sum(dim=1)
        term2 = ((mu2 - mu1).pow(2) / var2).sum(dim=1)
        term3 = (log_var2 - log_var1).sum(dim=1)

        if torch.isinf(term1.mean()) or torch.isinf(term2.mean()) or torch.isinf(term3.mean()):
            print("\nterm1: ", term1.mean())
            print(f"\nterm2: {term2.mean()}")
            print(f"\nterm3: {term3.mean()}")

            mask = torch.isinf(term1)
            inf_indices = torch.nonzero(mask)
            print("\nterm1 indices: ", term1[inf_indices])
            print(f"\nvar1 indices: {var1[inf_indices]} \nvar2 indices: {var2[inf_indices]}")
            print(f"\nlog_var1 indices: {log_var1[inf_indices]}, log_var2 indices: {log_var2[inf_indices]}")


        kl_div = 0.5 * (term1 + term2 + term3)
        return kl_div.mean()
        
    def iso_KL_divergence(self, mean, log_var):
        max_logvar = 10.0
        log_var = torch.clamp(log_var, max=max_logvar)
        return 0.5 * torch.mean(-1*(1+ log_var) + mean.pow(2) + log_var.exp())

    def cosine_sim(self, tensor1, tensor2):
        return torch.abs(F.cosine_similarity(tensor1, tensor2, dim=1)).mean()

    def byol_loss(self, x, y):
        # L2 normalization
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        loss = 2 - 2 * (x * y).sum(dim=-1)
        return loss.mean()
        

    def forward(self, o1, o2, t1, t2):

        KLD = 0
        kl_div = 0
        euclidean_distance = 0
        cosine_similarity = 0

        ##########################################ISOtrpoic gaussian
        KLD += self.iso_KL_divergence(o1[1], o1[2])
        KLD += self.iso_KL_divergence(o2[1], o2[2])
        KLD += self.iso_KL_divergence(t1[1], t1[2])
        KLD += self.iso_KL_divergence(t2[1], t2[2])
        ##########################################ISOtrpoic gaussian


        ########################################################KL Divergence 
        kl_div += self.kl_divergence(o1[1], o1[2], t2[1], t2[2])
        kl_div += self.kl_divergence(o2[1], o2[2], t1[1], t1[2])
        kl_div += self.kl_divergence(t2[1], t2[2], o1[1], o1[2])
        kl_div += self.kl_divergence(t1[1], t1[2], o2[1], o2[2])
        ########################################################KL Divergence 


        #######################################################################################Samples Loss
        # euclidean_distance += F.pairwise_distance(o1[4], t2[3], keepdim = True).squeeze().mean()
        # euclidean_distance += F.pairwise_distance(o2[4], t1[3], keepdim = True).squeeze().mean()
        
        euclidean_distance += self.byol_loss(o1[4], t2[3])
        euclidean_distance += self.byol_loss(o2[4], t1[3])
        # cosine_similarity += self.cosine_sim(o1[4], t2[3])
        # cosine_similarity += self.cosine_sim(o2[4], t1[3])
        # cosine_similarity += self.cosine_sim(t2[3], t1[3])
        # cosine_similarity += self.cosine_sim(o2[3], o1[3])

        #######################################################################################Samples Loss

        # weight1 = 300
        # weight2 = 1
        # weight3 = 30

        # KLD *= weight1 
        # kl_div *= weight2
        # euclidean_distance *= weight3

    
        total_loss = KLD +  euclidean_distance + kl_div
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(KLD, kl_div, cosine_similarity)
        return total_loss, [KLD, kl_div, euclidean_distance]