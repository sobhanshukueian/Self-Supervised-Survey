# MOCO VAR With Projection Layer

import torch.nn as nn
import torch
import torchvision.utils
import torchvision
import torch.nn.functional as F
from torchvision.models import resnet
from functools import partial
import torch.nn.init as init


from configs import model_config
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


class MOCOOOOOOO(nn.Module):
    def __init__(self, K=4000, m=0.99, T=0.1):
        super(MOCOOOOOOO, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = self.get_backbone()
        self.encoder_q.projection = self.get_mlp_block()
        self.encoder_q.mean = nn.Linear(model_config["EMBEDDING_SIZE"], model_config["PROJECTION_SIZE"])
        self.encoder_q.var = nn.Linear(model_config["EMBEDDING_SIZE"], model_config["PROJECTION_SIZE"])
        
        
        init.zeros_(self.encoder_q.var.weight)
        init.zeros_(self.encoder_q.mean.weight)


        self.encoder_k = self.get_backbone()
        self.encoder_k.projection = self.get_mlp_block()
        self.encoder_k.mean = nn.Linear(model_config["EMBEDDING_SIZE"], model_config["PROJECTION_SIZE"])
        self.encoder_k.var = nn.Linear(model_config["EMBEDDING_SIZE"], model_config["PROJECTION_SIZE"])

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(model_config["PROJECTION_SIZE"], K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.LeakyReLU = nn.LeakyReLU(0.2)



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

    def get_mlp_block(self):
        return nn.Sequential(
            nn.Linear(model_config["EMBEDDING_SIZE"], model_config["HIDDEN_SIZE"]),
            nn.BatchNorm1d(model_config["HIDDEN_SIZE"]),
            nn.ReLU(inplace=True),
            nn.Linear(model_config["HIDDEN_SIZE"], model_config["PROJECTION_SIZE"])
        )

    def get_backbone(self):
        backbone = ModelBase(arch="resnet50", feature_dim=model_config["EMBEDDING_SIZE"])
        return backbone

    def iso_kl(self, mean, log_var):
        return - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    def disentangled_contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized
        q_projected = self.encoder_q.projection(q)

        q_mean = self.encoder_q.mean(q)
        q_var = self.encoder_q.var(q)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)
            k_projected = self.encoder_k.projection(k)

            k_mean = self.encoder_k.mean(k)
            k_var = self.encoder_k.var(k)

        iso_kl_loss = self.iso_kl(q_mean, q_var)
        iso_kl_loss += self.iso_kl(k_mean, k_var)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # print(q_projected.size(), k_projected.size())
        l_pos = torch.einsum('nc,nc->n', [q_projected, k_projected]).unsqueeze(-1)

        # print(q_projected.size(), self.queue.clone().size())
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q_projected, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k_projected, iso_kl_loss

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