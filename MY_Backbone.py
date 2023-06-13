import math

import torch.nn as nn
import torch
import torchvision.utils
import torchvision
import torch.nn.functional as F

from configs import model_config
from Layers import *
import torch.nn.init as init

        
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
head=dict(
    type='Head',
    num_repeat=[3, 3],
    out_channels=[256, 512, 512, 512],
    args=[[3, 2, 1], [3, 2, 1]]
)

depth_multiple=0.5
width_multiple=0.25

channel_list_backbone = backbone['out_channels']
channel_list_neck = neck['out_channels']
channel_list_head = head['out_channels']

channels_backbone = [make_divisible(i * width_multiple, 8) for i in channel_list_backbone]
channels_neck = [make_divisible(i * width_multiple, 8) for i in channel_list_neck]
channels_head = [make_divisible(i * width_multiple, 8) for i in channel_list_head]



# channels_backbone =channel_list_backbone
# channels_neck = channel_list_neck 
# channels_head = channel_list_head 

backbone_args = backbone['args']
neck_args = neck['args']
head_args = head['args']


num_repeat_backbone = backbone['num_repeat']
num_repeat_neck = neck['num_repeat']
num_repeat_head = head['num_repeat']


repeat_backbone = [(max(round(i * depth_multiple), 1) if i > 1 else i) for i in num_repeat_backbone]
repeat_neck = [(max(round(i * depth_multiple), 1) if i > 1 else i) for i in num_repeat_neck]
repeat_head = [(max(round(i * depth_multiple), 1) if i > 1 else i) for i in num_repeat_head]


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
            C3(ch_list[7], ch_list[8], bottle_neck_num=num_repeat[3]),
            # SPPF
            SPPF(ch_list[8], ch_list[9])
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
        self.conv_block1 = Conv2D(in_ch=in_ch, out_ch=ch_list[0], kernel=args[0][0], stride=args[0][1],
                                  padding=args[0][2])
        self.conv_block2 = Transpose(ch_list[0], ch_list[1])
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
        # print(out_conv_block_1.size())
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


class Head(nn.Module):
    def __init__(self, in_ch=64, ch_list=channels_head, num_repeat=repeat_head, args=head_args):
        super().__init__()
        assert ch_list is not None
        assert num_repeat is not None
        assert args is not None
        self.conv_block1 = Conv2D(in_ch=in_ch, out_ch=ch_list[0], kernel=args[0][0], stride=args[0][1], padding=args[0][2])
        self.conv_block2 = C3(in_ch=ch_list[0] * 2, out_ch=ch_list[1], shortcut=False, bottle_neck_num=num_repeat[0])
        self.conv_block3 = Conv2D(in_ch=ch_list[1], out_ch=ch_list[2], kernel=args[1][0], stride=args[1][1], padding=args[1][2])
        self.conv_block4 = C3(in_ch=ch_list[2] * 2, out_ch=ch_list[3], shortcut=False, bottle_neck_num=num_repeat[1])

    def forward(self, x):
        (x2, x1, x0) = x
        outputs = [x0]
        out_conv_block_1 = self.conv_block1(x0)
        out_concat_1 = torch.cat([out_conv_block_1, x1], 1)
        out_conv_block_2 = self.conv_block2(out_concat_1)
        outputs.append(out_conv_block_2)
        out_conv_block_3 = self.conv_block3(out_conv_block_2)
        # print(out_conv_block_3.size())
        # print(x2.size())


        out_concat_2 = torch.cat([out_conv_block_3, x2], 1)
        # print(out_concat_2.size())
        out_conv_block_4 = self.conv_block4(out_concat_2)
        outputs.append(out_conv_block_4)
        return list(outputs)


#create the Siamese Neural Network
class MyBackbone(nn.Module):

    def __init__(self):
        super(MyBackbone, self).__init__()
        # self.online = self.get_rep_and_proj(in_features, embedding_size, hidden_size, batch_norm_mlp)
        self.online =  Backbone()
        self.online.neck = Neck()
        self.online.head = Head()

        self.ss1 = Conv2D(in_ch=128, out_ch=256, kernel=2, stride=1, padding=0)
        self.ss0 = Conv2D(in_ch=64, out_ch=256, kernel=4, stride=1, padding=0)
        self.ss2 = Conv2D(in_ch=128, out_ch=512, kernel=1, stride=1, padding=0)

        self.fc = self.get_mlp_block(in_ch=1024)


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

    def forward(self, x1, x2=None, return_embedding=False):
        x1 = self.online(x1)
        # print(x1[0].size(), x1[1].size(), x1[2].size())
        x1 = self.online.neck(x1)
        # print(x1[0].size(), x1[1].size(), x1[2].size())
        x1 = self.online.head(x1)
        # print(x1[0].squeeze().size(), x1[1].squeeze().size(), x1[2].squeeze().size())
        x12 = self.ss2(x1[2])
        x11 = self.ss1(x1[1])
        x10 = self.ss0(x1[0])


        # print(x10.squeeze().size(), x11.squeeze().size(), x12.squeeze().size())

        sss = torch.cat([x10, x11, x12], 1).squeeze()

        sss = self.fc(sss)
        return sss


#create the Siamese Neural Network
class GaussianProjection(nn.Module):

    def __init__(self):
        super(GaussianProjection, self).__init__()
        self.projection = self.get_mlp_block(model_config["EMBEDDING_SIZE"])
        self.mean = nn.Linear(model_config["PROJECTION_SIZE"], model_config["PROJECTION_SIZE"])
        self.var = nn.Linear(model_config["PROJECTION_SIZE"], model_config["PROJECTION_SIZE"])

        init.zeros_(self.var.weight)
        init.zeros_(self.mean.weight)


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

    def forward(self, x1):
        x1 = self.projection(x1)
        x1_mean = self.mean(x1)
        x1_var = self.var(x1)

        return x1_mean, x1_var


# temp1 = torch.rand((10, 3, 32, 32))
temp_model = MyBackbone()
# ress = temp_model(temp1)
# print(ress.size())
total_params = 0

for name, parameter in temp_model.named_parameters():
    if not parameter.requires_grad: continue
    params = parameter.numel()
    # table.add_row([name, params])
    total_params+=params

# print(total_params)