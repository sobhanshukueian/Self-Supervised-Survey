# MOCO VAR with projection layer

import torch.nn as nn
import torch
import torchvision.utils
import torchvision
import torch.nn.functional as F
from torchvision.models import resnet
from functools import partial
import torch.nn.init as init
import torchvision.models as torchvision_models


from configs import model_config
from MY_Backbone import MyBackbone, GaussianProjection
import copy

class MOCO4(nn.Module):
    def __init__(self, K=4000, m=0.99, T=0.1):
        super(MOCO4, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = partial(torchvision_models.__dict__["resnet50"], zero_init_residual=True)(num_classes=model_config["EMBEDDING_SIZE"])
        self.encoder_k = partial(torchvision_models.__dict__["resnet50"], zero_init_residual=True)(num_classes=model_config["EMBEDDING_SIZE"])

        hidden_dim = self.encoder_q.fc.weight.shape[1]
        del self.encoder_q.fc, self.encoder_k.fc # remove original fc layer

        # print(hidden_dim)
        self.encoder_q.fc = self.get_mlp_block(hidden_dim)
        self.encoder_k.fc = self.get_mlp_block(hidden_dim)


        self.encoder_q_gaussian = GaussianProjection()
        self.predictor = self.get_mlp_block(model_config["PROJECTION_SIZE"])

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(model_config["PROJECTION_SIZE"], K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    def get_mlp_block(self, in_ch):
        return nn.Sequential(
            nn.Linear(in_ch, model_config["HIDDEN_SIZE"]),
            nn.BatchNorm1d(model_config["HIDDEN_SIZE"]),
            nn.ReLU(inplace=True),
            nn.Linear(model_config["HIDDEN_SIZE"], model_config["HIDDEN_SIZE"]),
            nn.BatchNorm1d(model_config["HIDDEN_SIZE"]),
            nn.ReLU(inplace=True),
            nn.Linear(model_config["HIDDEN_SIZE"], model_config["PROJECTION_SIZE"])
        )

    @torch.no_grad()
    def  _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def iso_kl(self, mean, log_var):
        return - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    def disentangled_contrastive_loss(self, im_q, im_k):
        # compute query features
        # print(self.encoder_q(im_q).size())
        q = self.encoder_q(im_q)
        
        q_predicted = self.predictor(q)  # queries: NxC
        q_predicted = nn.functional.normalize(q_predicted, dim=1)  # already normalized

        q_mean, q_var = self.encoder_q_gaussian(q)
        iso_kl_loss = self.iso_kl(q_mean, q_var)


        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # print(q_projected.size(), k_projected.size())
        l_pos = torch.einsum('nc,nc->n', [q_predicted, k]).unsqueeze(-1)

        # print(q_projected.size(), self.queue.clone().size())
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q_predicted, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k, iso_kl_loss

    def forward(self, im1, im2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        loss_12, q1, k2, iso_kl_loss1 = self.disentangled_contrastive_loss(im1, im2)
        loss_21, q2, k1, iso_kl_loss2 = self.disentangled_contrastive_loss(im2, im1)

        iso_kl_loss1 *= 0.001
        iso_kl_loss2 *= 0.001
        iso_kl_total = iso_kl_loss1 + iso_kl_loss2

        loss = loss_12 + loss_21 + iso_kl_total
        k = torch.cat([k1, k2], dim=0)

        self._dequeue_and_enqueue(k)

        return (q1, q2), loss, [loss_12, loss_21, iso_kl_total]

# create model
# model = ModelMoCo().cuda()
    
# print(model.encoder_q)