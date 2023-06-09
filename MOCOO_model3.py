import torch.nn as nn
import torch
import torchvision.utils
import torchvision
import torch.nn.functional as F
from torchvision.models import resnet
from functools import partial
import torch.nn.init as init


from configs import model_config
from utils import update_momentum, initialize_keys, weights_init_zero
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
        self.T = T

        # create the encoders
        self.q_backbone = self.get_backbone()
        self.q_projection = self.get_mlp_block()
        self.q_mean = self.get_mlp_block()
        self.q_var = self.get_mlp_block()
        
        # init.zeros_(self.encoder_q.var.weight)
        # init.zeros_(self.encoder_q.mean.weight)
        # weights_init_zero(q_mean)
        # weights_init_zero(q_var)
        # print()

        self.k_backbone = self.get_backbone()
        self.k_projection = self.get_mlp_block()
        self.k_mean = self.get_mlp_block()
        self.k_var = self.get_mlp_block()

        initialize_keys(self.q_backbone, self.k_backbone)
        initialize_keys(self.q_projection, self.k_projection)
        initialize_keys(self.q_mean, self.k_mean)
        initialize_keys(self.q_var, self.k_var)

        # create the queue
        self.register_buffer("queue", torch.randn(model_config["EMBEDDING_SIZE"], K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.LeakyReLU = nn.LeakyReLU(0.2)


    @torch.no_grad()
    def  _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """ 
        update_momentum(self.q_backbone, self.k_backbone)
        update_momentum(self.q_projection, self.k_projection)
        update_momentum(self.q_mean, self.k_mean)
        update_momentum(self.q_var, self.k_var)


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
            nn.Linear(model_config["HIDDEN_SIZE"], model_config["EMBEDDING_SIZE"])
        )

    def get_backbone(self):
        backbone = ModelBase(arch="resnet50", feature_dim=model_config["EMBEDDING_SIZE"])
        return backbone

    def iso_kl(self, mean, log_var):
        return - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    def disentangled_contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.q_backbone(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        q_projected = self.q_projection(q) #project

        #Disentanglement
        q_mean = self.q_mean(self.LeakyReLU(q))
        q_var = self.q_var(self.LeakyReLU(q))

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.k_backbone(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized
            k_projected = self.k_projection(k) # project k

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)
            
            k_mean = self.k_mean(self.LeakyReLU(k))
            k_var = self.k_var(self.LeakyReLU(k))


        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q_projected, k_projected]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q_projected, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        iso_kl_loss = self.iso_kl(q_mean, q_var)
        iso_kl_loss += self.iso_kl(k_mean, k_var)

        return loss, iso_kl_loss, q, k_projected

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
        loss_12, iso_kl_loss12, q1, k2 = self.disentangled_contrastive_loss(im1, im2)
        loss_21, iso_kl_loss21, q2, k1 = self.disentangled_contrastive_loss(im2, im1)

        iso_kl_loss12 *= 0.001
        iso_kl_loss21 *= 0.001
        iso_kl_total = iso_kl_loss12 + iso_kl_loss21

        loss = loss_12 + loss_21 + iso_kl_total
        k = torch.cat([k1, k2], dim=0)

        self._dequeue_and_enqueue(k)

        return (q1, k1), loss, [loss_12, loss_21, iso_kl_total]

# create model
model = MOCOOOOOOO().cuda()
model._momentum_update_key_encoder()
# print(model.encoder_q)