import torch
import matplotlib
import matplotlib.pyplot as plt
import json
import os.path as osp
import os
import shutil
from copy import deepcopy
from prettytable import PrettyTable
from torch.optim.lr_scheduler import _LRScheduler
import math
import numpy as np
import logging
from PIL import Image
import torchvision.transforms.functional as TF
import random

from configs import model_config
import torch.nn as nn

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]



class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def get_color(idx):
    return list(matplotlib.colors.cnames.keys())[idx]
  
def get_colors(idxs):
    res = []
    for idx in idxs:
      res.append(get_color(idx))
    return res 

class EMA():
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.alpha + (1 - self.alpha) * new

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=10):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # linear layer
        return self.linear(x)

def count_parameters(logger, model, conf):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    logger.warning(f"Total Trainable Params: {total_params}")
    conf["Parameter_size"] = total_params
    return conf

def save(conf, save_dir, model_name, model, epoch, val_loss, best_loss, optimizer):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    # create config object
    conf = json.dumps(conf)
    f = open(save_dir + "/config.json","w")
    f.write(conf)
    f.close()

    # save model
    save_ckpt_dir = osp.join(save_dir, 'weights')
    if not osp.exists(save_ckpt_dir):
        os.makedirs(save_ckpt_dir)
    filename = osp.join(save_ckpt_dir,'last.pt')

    # save ckpt
    ckpt = {
            'model': deepcopy(model).half(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            }
    torch.save(ckpt, filename)
    if val_loss == best_loss:
        best_filename = osp.join(save_ckpt_dir, 'best_{}.pt'.format(model_name, epoch))
        if osp.exists(best_filename):
            os.remove(best_filename)
        shutil.copyfile(filename, best_filename)

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def compute_acc(predicted, labels):
    # print(predicted.size(), labels.size())
    predicted = np.argmax(predicted, 1)  
    correct = (predicted == labels).sum().item() 
    total = len(labels)
    return (100 * correct / total)


    
def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Decay the learning rate based on schedule"""
    lr = learning_rate
    warmup_epoch = model_config["WARM_UP"]
    # cosine lr schedule
    warmup_lr_schedule = np.linspace(0, lr, warmup_epoch)

    if epoch < warmup_epoch:
        lr = warmup_lr_schedule[epoch]
    else:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epoch) / (model_config["EPOCHS"] - warmup_epoch)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f'epoch:{epoch}, lr: {lr}')

def adjust_loss_weights(epoch):
    return 0.5 * ( 1. + math.cos(0.1*epoch)) * 100000

def update_momentum(updated_params, updatable_params, m=0.99):
    for updated, updatable in zip(updated_params.parameters(), updatable_params.parameters()):
        updatable.data = updatable.data * m + updated.data * (1. - m)

def initialize_keys(query, key):
    for param_q, param_k in zip(query.parameters(), key.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient


def weights_init_zero(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.weight)
        m.bias.data.fill_(0)

def set_logging(save_dir, name=None):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    logger = logging.getLogger(name)
    # Create handlers
    handler = logging.StreamHandler()
    handler = logging.FileHandler(f'{save_dir}/{name}.log')
    format = logging.Formatter('%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M')
    handler.setFormatter(format)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.warning(f"{name} LOGGER")
    return logger

class random_mask(object):
    def __init__(self, output_size, mask_size, p):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.mask_size = mask_size
        self.p = p

    def __call__(self, image):
        """
        Apply random small black squares, circles, and rectangles to the input image.

        Args:
            image (PIL Image or Tensor): Input image.
            target_size (tuple): Desired size of the image. Default is (256, 256).
            mask_size (int): Size of the small masks. Default is 10.
            num_masks (int): Number of masks to apply. Default is 10.

        Returns:
            PIL Image or Tensor: Augmented image with random small masks applied.
        """

        if random.random() < self.p:
            return image

        num_masks = int(abs(random.gauss(50, 50)))

        # Convert the image to RGB if it has an alpha channel
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Convert the image to a Tensor
        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)

        # Generate random mask positions
        image_height, image_width = image.size(1), image.size(2)
        mask_positions = np.random.randint(low=0, high=min(image_height, image_width) - self.mask_size, size=(num_masks, 2))

        # Apply the masks to the image
        augmented_image = image.clone()
        for position in mask_positions:
            x1, y1 = position

            # Generate random mask shape
            mask_shape = np.random.choice(['square', 'rectangle'])

            if mask_shape == 'square':
                x2, y2 = x1 + self.mask_size, y1 + self.mask_size
                augmented_image[:, y1:y2, x1:x2] = 0
            elif mask_shape == 'rectangle':
                x2, y2 = x1 + self.mask_size * 2, y1 + self.mask_size
                augmented_image[:, y1:y2, x1:x2] = 0

        # Convert the augmented image back to a PIL Image if necessary
        if not isinstance(image, torch.Tensor):
            augmented_image = TF.to_pil_image(augmented_image)

        return augmented_image