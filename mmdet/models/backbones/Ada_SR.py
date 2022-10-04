import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.functional import interpolate, conv2d
from ..builder import BACKBONES
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import BaseModule


class Res_block(nn.Module):
    def __init__(self, in_c=48, mid_c=16):
        super(Res_block, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(in_c, mid_c, 3, 1, 1, bias=True), nn.ReLU(),
                                        nn.Conv2d(mid_c, in_c, 3, 1, 1, bias=True))

    def forward(self, x):
        add = x
        x = self.conv_block(x)
        x = x + add
        return x

@BACKBONES.register_module()
class Adaptive_SR(nn.Module):
    def __init__(self, out_size=(128, 128), in_c=64, out_c=3, ps_rate=4):
        super(Adaptive_SR, self).__init__()
        # channel number before pixel shuffle
        mid_c = out_c*ps_rate*ps_rate
        self.out_size = out_size
        self.conv_init = nn.Conv2d(in_c, mid_c, 3, 1, 1, bias=True)
        self.res_1 = Res_block(48,16)
        self.res_2 = Res_block(48,16)

        self.ps = nn.PixelShuffle(4)

    def forward(self, x):
        x1 = self.conv_init(x)
        add = F.interpolate(x1, self.out_size, mode='bilinear')
        x1 = self.res_1(x1)
        x1 = self.res_2(x1)
        x1 = F.interpolate(x1, self.out_size, mode='bilinear')
        #print(x1.shape)
        x_out = x1 + add
        x_out = self.ps(x_out)
        #print(x_out.shape)
        return x_out


