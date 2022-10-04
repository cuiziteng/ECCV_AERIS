import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
#from torch.nn.functional import interpolate, conv2d
from ..builder import BACKBONES
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

def conv_block(in_channels, filters, kernel_size, strides, padding , mode='cba'):
    
    conv = nn.Conv2d(in_channels, filters, kernel_size, strides, padding, bias=False)
    bn = nn.BatchNorm2d(filters)
    act = nn.LeakyReLU(0.2)

    if mode == 'cba':  
        return nn.Sequential(conv, bn, act)
    elif mode == 'cb':
        return nn.Sequential(conv, bn)
    elif mode == 'cab':
        return nn.Sequential(conv, act, bn)
    elif mode == 'ca':
        return nn.Sequential(conv, act)
    elif mode == 'c':
        return conv

class Res_block(nn.Module):
    def __init__(self, channels=64):
        super(Res_block, self).__init__()
        # first with bn
        self.conv1 = conv_block(channels, channels, 3, 1, 1, 'cba')
        # second without activation
        self.conv2 = conv_block(channels, channels, 3, 1, 1, 'cb')

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.conv2(out)
        out += x
        return out

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=256, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)  
        return x

@BACKBONES.register_module()
class ARRD_FPN(BaseModule):
    def __init__(self, 
        in_channel=256,
        mid_channel=64,
        mode='bilinear',
        ps_up=True,
        # scale_factor = s,
        init_cfg=dict(type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(ARRD_FPN, self).__init__(init_cfg)
        '''
        mode (str) – algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' 
        | 'bicubic' | 'trilinear' | 'area'. Default: 'bilinear'
        '''
        self.mode = mode
        self.conv_init = conv_block(in_channel, mid_channel, 3, 1, 1, 'cab')
        self.res_block = Res_block(mid_channel)
        
        #self.dropout = nn.Dropout2d(0.1)
        self.ps_up = ps_up
        if self.ps_up:
            self.conv_final = conv_block(mid_channel, 48, 3, 1, 1, 'c')
            self.ps = nn.PixelShuffle(4)
        else:
            self.conv_final = nn.Sequential(conv_block(mid_channel, 48, 3, 1, 1, 'ca'), 
                                        conv_block(48, 3, 3, 1, 1, 'c'))
    
    def forward(self, x, s):
        x = self.conv_init(x[0])    # only extract one level features
        #print(x.shape)
        if self.pre_up_rate != 1:
            x = self.pre_up_conv(x)

        res = F.interpolate(x, scale_factor=s, mode=self.mode, align_corners=False)    # up-scale
        x = self.res_block(x)
        x = F.interpolate(x, scale_factor=s, mode=self.mode, align_corners=False)      # down-scale

        x += res  # short cut 1
        x_final = self.conv_final(x)
        #x_final = self.dropout(x_final)
        if self.ps_up:
            x_final = self.ps(x_final)
        

        return x_final