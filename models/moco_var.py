# MOCO VAR2 without projection layer

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
from models.variation import GaussianProjection
from models.layers import ModelBase
from models.backbone import MyBackbone
import copy


class MOCO_VAR_MODEL(nn.Module):
    def __init__(self, K=4096, m=0.99, T=0.1, arch='resnet18'):
        super(MOCO_VAR_MODEL, self).__init__()

        self.K = K
        self.m = m
        self.T = T


        if "resnet" in model_config["Backbone"]:
            # create the encoders
            self.encoder = ModelBase(feature_dim=model_config["EMBEDDING_SIZE"], arch=arch)
            self.encoder_k = ModelBase(feature_dim=model_config["EMBEDDING_SIZE"], arch=arch)
        else : 
            self.encoder = MyBackbone()
            self.encoder_k = MyBackbone()

        self.encoder_q_gaussian = GaussianProjection()
        self.encoder_k_gaussian = GaussianProjection()

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        for param_q, param_k in zip(self.encoder_q_gaussian.parameters(), self.encoder_k_gaussian.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(model_config["PROJECTION_SIZE"], K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def  _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.encoder_q_gaussian.parameters(), self.encoder_k_gaussian.parameters()):
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
        q = self.encoder(im_q)
        q = nn.functional.normalize(q, dim=1)  # already normalized

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
            k_mean, k_var = self.encoder_k_gaussian(k)


        mean = torch.cat([q_mean, k_mean], dim=0)
        cos_sim = F.cosine_similarity(mean[:,None,:], mean[None,:,:], dim=-1)
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        cos_sim = cos_sim #/ self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        loss_gaussian = nll.mean()


        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k, iso_kl_loss, loss_gaussian

    def forward(self, im1, im2, train):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        if train:
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()

        # compute loss
        loss_12, q1, k2, iso_kl_loss1, loss_gaussian1 = self.disentangled_contrastive_loss(im1, im2)
        loss_21, q2, k1, iso_kl_loss2, loss_gaussian2 = self.disentangled_contrastive_loss(im2, im1)

        loss_gaussian_total = loss_gaussian1 + loss_gaussian2

        iso_kl_loss1 *= 0.001
        iso_kl_loss2 *= 0.001
        iso_kl_total = iso_kl_loss1 + iso_kl_loss2

        loss_total = loss_12 + loss_21

        loss = loss_total + iso_kl_total + loss_gaussian_total
        k = torch.cat([k1, k2], dim=0)

        if train:
            self._dequeue_and_enqueue(k)

        # print("loss Guassian mean: {}\nloss iso KLD: {}\nloss contrastive: {}".format(loss_gaussian_total, iso_kl_total, loss_total))


        return (q1, q2), loss, [loss_total, loss_gaussian_total, iso_kl_total]

# create model
# model = ModelMoCo().cuda()
    
# print(model.encoder_q)