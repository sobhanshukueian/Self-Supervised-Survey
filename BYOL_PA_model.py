import math

import torch.nn as nn
import torch
import torchvision.utils
import torchvision
import torch.nn.functional as F


import copy
from utils import EMA

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


        import math
backbone=dict(
    type='Backbone',
    num_repeat=[3, 6, 9, 3],
    out_channels=[64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024],
    args=[[6, 2, 2], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1]]
)

neck=dict(
    type='Neck',
    num_repeat=[3, 3],
    out_channels=[512, 512, 512, 256, 256, 256],
    args=[[1, 1, 0], [1, 1, 0]]
)

depth_multiple=0.5
width_multiple=0.25
channel_list_backbone = backbone['out_channels']
channel_list_neck = neck['out_channels']
channels_backbone = [make_divisible(i * width_multiple, 8) for i in channel_list_backbone]
channels_neck = [make_divisible(i * width_multiple, 8) for i in channel_list_neck]

backbone_args = backbone['args']
neck_args = neck['args']

num_repeat_backbone = backbone['num_repeat']
num_repeat_neck = neck['num_repeat']

repeat_backbone = [(max(round(i * depth_multiple), 1) if i > 1 else i) for i in num_repeat_backbone]
repeat_neck = [(max(round(i * depth_multiple), 1) if i > 1 else i) for i in num_repeat_neck]




class Backbone(nn.Module):
    def __init__(self, in_ch=3, ch_list=channels_backbone, num_repeat=repeat_backbone, args=backbone_args):
        super().__init__()
        assert ch_list is not None
        assert num_repeat is not None
        self.conv_block1 = nn.Sequential(
            # P1 / P2
            Conv2D(in_ch, ch_list[0], kernel=args[0][0], stride=args[0][1], padding=args[0][2]),
            Conv2D(ch_list[0], ch_list[1], kernel=args[1][0], stride=args[1][1], padding=args[1][2]),
            #  C3
            C3(ch_list[1], ch_list[2], bottle_neck_num=num_repeat[0]),
            # P3
            Conv2D(ch_list[2], ch_list[3], kernel=args[2][0], stride=args[2][1], padding=args[2][2]),
            # C3
            C3(ch_list[3], ch_list[4], bottle_neck_num=num_repeat[1])
        )
        self.conv_block2 = nn.Sequential(
            # P4
            Conv2D(ch_list[4], ch_list[5], kernel=args[3][0], stride=args[3][1], padding=args[3][2]),
            # C3
            C3(ch_list[5], ch_list[6], bottle_neck_num=num_repeat[2])
        )
        self.conv_block3 = nn.Sequential(
            # P5
            Conv2D(ch_list[6], ch_list[7], kernel=args[4][0], stride=args[4][1], padding=args[4][2]),
            # C3
            C3(ch_list[7], ch_list[8], bottle_neck_num=num_repeat[3])
        )

    def forward(self, x):
        outputs = []
        out_conv_block_1 = self.conv_block1(x)
        outputs.append(out_conv_block_1)
        out_conv_block_2 = self.conv_block2(out_conv_block_1)
        outputs.append(out_conv_block_2)
        out_conv_block_3 = self.conv_block3(out_conv_block_2)
        outputs.append(out_conv_block_3)
        return tuple(outputs)


class Neck(nn.Module):
    def __init__(self, in_ch=256, ch_list=channels_neck, num_repeat=repeat_neck, args=neck_args):
        super().__init__()
        assert ch_list is not None
        assert num_repeat is not None
        assert args is not None
        self.conv_block1 = Conv2D(in_ch=in_ch, out_ch=ch_list[0]*4, kernel=args[0][0], stride=args[0][1],
                                  padding=args[0][2])
        self.conv_block2 = Transpose(ch_list[0]*4, ch_list[1])
        self.conv_block3 = nn.Sequential(
            C3(ch_list[1] * 2, ch_list[2], shortcut=False, bottle_neck_num=num_repeat[0]),
            Conv2D(ch_list[2], ch_list[3], kernel=args[1][0], stride=args[1][1], padding=args[1][2])
        )
        self.conv_block4 = Transpose(ch_list[3], ch_list[4])
        self.conv_block5 = C3(ch_list[4] * 2, ch_list[5], shortcut=False, bottle_neck_num=num_repeat[1])

    def forward(self, x):
        (x2, x1, x0) = x
        outputs = []
        out_conv_block_1 = self.conv_block1(x0)
        outputs.append(out_conv_block_1)
        out_conv_block_2 = self.conv_block2(out_conv_block_1)
        out_concat_1 = torch.cat([out_conv_block_2, x1], 1)

        out_conv_block_3 = self.conv_block3(out_concat_1)
        outputs.append(out_conv_block_3)
        out_conv_block_4 = self.conv_block4(out_conv_block_3)
        out_concat_2 = torch.cat([out_conv_block_4, x2], 1)

        out_conv_block_5 = self.conv_block5(out_concat_2)
        outputs.append(out_conv_block_5)
        
        return tuple(outputs)

#create the Siamese Neural Network
class BYOLPANetwork(nn.Module):

    def __init__(self, in_features=512, hidden_size=4096, embedding_size=256, projection_size=512, projection_hidden_size=2048, batch_norm_mlp=True):
        super(BYOLPANetwork, self).__init__()
        # self.online = self.get_rep_and_proj(in_features, embedding_size, hidden_size, batch_norm_mlp)
        self.online =  Backbone()
        self.online.neck = Neck()
        self.predictor = self.get_cnn_block(projection_size, projection_size, projection_hidden_size)
        self.target = self.get_target()
        self.ema = EMA(0.99)
    
    @torch.no_grad()
    def get_target(self):
        return copy.deepcopy(self.online)

    def get_cnn_block(self, dim, embedding_size=256, hidden_size=2048, batch_norm_mlp=False):
        norm = nn.BatchNorm1d(hidden_size) #if batch_norm_mlp else nn.Identity()
        return nn.Sequential(
            nn.Linear(dim, hidden_size),
            norm,
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, embedding_size)
        )

    def get_rep_and_proj(self, in_features, embedding_size, hidden_size, batch_norm_mlp):
        self.backbone = torchvision.models.resnet18(num_classes=hidden_size)  # Output of last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.backbone.fc = nn.Sequential(
            self.backbone.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, in_features)
        )
        proj = self.get_cnn_block(in_features, embedding_size, hidden_size=hidden_size, batch_norm_mlp=batch_norm_mlp)
        return nn.Sequential(self.backbone, proj)

    @torch.no_grad()
    def update_moving_average(self):
        for online_params, target_params in zip(self.online.parameters(), self.target.parameters()):
            old_weight, up_weight = target_params.data, online_params.data
            target_params.data = self.ema.update_average(old_weight, up_weight)


    def forward(self, x1, x2=None, return_embedding=False):
        if return_embedding or (x2 is None):
            x1 = self.online(x1)
            x1 = self.online.neck(x1)[0].squeeze()
            return x1

        # online projections: backbone + MLP projection
        x1_1 = self.online(x1)
        x1_1 = self.online.neck(x1_1)[0].squeeze()
        # print(x1_1.size())
        x1_2 = self.online(x2)
        x1_2 = self.online.neck(x1_2)[0].squeeze()
        # x1_2 = self.online(x2)

        # additional online's MLP head called predictor
        x1_1_pred = self.predictor(x1_1)
        x1_2_pred = self.predictor(x1_2)

        with torch.no_grad():
            # teacher processes the images and makes projections: backbone + MLP
            x2_1 = self.target(x1)
            x2_1 = self.target.neck(x2_1)[0].squeeze().detach()
            x2_2 = self.target(x2)
            x2_2 = self.target.neck(x2_2)[0].squeeze().detach()

        return x1_1_pred, x1_2_pred, x2_1, x2_2

def byol_loss(x, y):
    # L2 normalization
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    loss = 2 - 2 * (x * y).sum(dim=-1)
    return loss        

# temp1 = torch.rand((6, 3, 32, 32))
# temp2 = torch.rand((6, 3, 32, 32))
# temp_model = BYOLPANetwork()
# res = temp_model(temp1, temp2)
# print(res[0].size(), res[1].size(), res[2].size(), res[3].size())