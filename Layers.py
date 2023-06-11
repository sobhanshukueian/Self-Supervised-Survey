import math

import torch.nn as nn
import torch
import torchvision.utils
import torchvision
import torch.nn.functional as F


import copy

def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor

def auto_pad(k, p=None):
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

    
class Transpose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        return self.upsample_transpose(x)


class Conv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=1, stride=1, padding=None, groups=1, hasActivation=True):
        super(Conv2D, self).__init__()
        self.conv2D = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=auto_pad(kernel, padding),
                                groups=groups, bias=False)
        self.BN = nn.BatchNorm2d(out_ch)
        if hasActivation:
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x = self.conv2D(x)
        x = self.BN(x)
        x = self.activation(x)
        return x

    def forward_fuse(self, x):
        x = self.conv2D(x)
        x = self.activation(x)  
        return x


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.dimension = dimension

    def forward(self, x):
        return torch.cat(x, self.dimension)


class DWConv(Conv2D):
    # convert simple conv to depth wise conv by changing groups parameter
    def __init__(self, in_ch, out_ch, kernel=1, stride=1, hasActivation=True):
        super().__init__(in_ch, out_ch, kernel, stride, groups=math.gcd(in_ch, out_ch), hasActivation=hasActivation)


class Bottleneck(nn.Module):
    # 2 types of bottleneck -> one with addition, one not
    def __init__(self, in_ch, out_ch, shortcut=True, groups=1, e=0.5):
        super().__init__()
        mid_ch = int(out_ch * e)  # hidden channels
        self.conv1 = Conv2D(in_ch, mid_ch, 1, 1)
        self.conv2 = Conv2D(mid_ch, out_ch, 3, 1, groups=groups)
        self.add = shortcut and in_ch == out_ch

    def forward(self, x):
        if self.add:
            mid = self.conv1(x)
            out = self.conv2(mid)
            return x + out
        else:
            mid = self.conv1(x)
            out = self.conv2(mid)
            return out


class C3(nn.Module):
    def __init__(self, in_ch, out_ch, bottle_neck_num=1, shortcut=True, groups=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        mid_ch = int(out_ch * e)  # hidden channels
        self.conv1 = Conv2D(in_ch, mid_ch, 1, 1)
        self.conv2 = Conv2D(in_ch, mid_ch, 1, 1)
        self.conv3 = Conv2D(2 * mid_ch, out_ch, 1)
        self.bottleneck = nn.Sequential(
            *(Bottleneck(mid_ch, mid_ch, shortcut, groups, e=1.0) for _ in range(bottle_neck_num))
        )

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_bottleneck = self.bottleneck(out_conv1)
        out_concat = torch.cat((out_bottleneck, self.conv2(x)), dim=1)
        out = self.conv3(out_concat)
        return out

class SPPF(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5):
        super().__init__()
        mid_ch = in_ch // 2
        self.conv1 = Conv2D(in_ch, mid_ch, 1, 1)
        self.conv2 = Conv2D(mid_ch * 4, out_ch, 1, 1)
        self.max_pool = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel // 2)

    def forward(self, x):
        x = self.conv1(x)
        # with warnings.catch_warnings():
            # warnings.simplefilter('ignore')
        out_max_pool_1 = self.max_pool(x)
        out_max_pool_2 = self.max_pool(out_max_pool_1)
        out_max_pool_3 = self.max_pool(out_max_pool_2)
        out_concat = torch.cat([x, out_max_pool_1, out_max_pool_2, out_max_pool_3], dim=1)
        out = self.conv2(out_concat)
        return out
