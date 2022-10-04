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
        self.conv1 = conv_block(channels, channels, 3, 1, 1, 'ca')
        # second without activation
        self.conv2 = conv_block(channels, channels, 3, 1, 1, 'c')

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
class ARRD_Single_Level(BaseModule):
    def __init__(self, 
        in_channel=64,
        mid_channel=48,
        #out_channel=512,
        pre_up_rate=2, 
        mode='bilinear',
        ps_up=True,
        # scale_factor = s,
        init_cfg=None):
        super(ARRD_Single_Level, self).__init__(init_cfg)
        '''
        mode (str) – algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' 
        | 'bicubic' | 'trilinear' | 'area'. Default: 'bilinear'
        '''
        self.mode = mode 
        # first conv without bn
        self.pre_up_rate = pre_up_rate
        self.conv_init = conv_block(in_channel, mid_channel, 3, 1, 1, 'cba')
        if self.pre_up_rate != 1:
            self.pre_up_conv = nn.ConvTranspose2d(mid_channel, mid_channel, 3, stride=pre_up_rate, padding=1, output_padding=1)
        # self.conv_1 = conv_block(3, 16, 7, 1, 3, 'ca')
        self.res_block = Res_block(mid_channel)
        
        #self.dropout = nn.Dropout2d(0.1)
        self.ps_up = ps_up
        if self.ps_up:
            self.conv_final = conv_block(mid_channel, 48, 3, 1, 1, 'ca')
            self.ps = nn.PixelShuffle(4)
        else:
            self.conv_final = nn.Sequential(conv_block(mid_channel, 48, 3, 1, 1, 'ca'), 
                                            conv_block(48, 3, 3, 1, 1, 'c'))
    
    def forward(self, x, s):
        #print(x.shape)
        x = self.conv_init(x)    # only extract one level features
        
        if self.pre_up_rate != 1:
            x = self.pre_up_conv(x)

        res = F.interpolate(x, scale_factor=s, mode=self.mode, align_corners=False)    # up-scale
        x = self.res_block(x)
        x = F.interpolate(x, scale_factor=s, mode=self.mode, align_corners=False)      # down-scale

        x += res  # short cut 1
        x_final = self.conv_final(x)
        if self.ps_up:
            x_final = self.ps(x_final)
        

        return x_final


# For FPN-like structure
@BACKBONES.register_module()
class ARRD_Multi_Level(BaseModule):
    def __init__(self, 
        in_channel=256,
        mid_channel=128,
        #embedding_dim=1024, 
        mode='bilinear',
        input_number=4, # Number of level features use for reconstruction
        init_cfg=dict(type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(ARRD_Multi_Level, self).__init__(init_cfg)
        '''
        mode (str) – algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' 
        | 'bicubic' | 'trilinear' | 'area'. Default: 'bilinear'
        '''
        self.mode = mode 
        self.input_number = input_number
        ## Level of features to use
        if self.input_number >= 4:
            #self.linear_c4 = MLP(input_dim=in_channel, embed_dim=embedding_dim)
            self.linear_c4 = MLP(input_dim=in_channel, embed_dim=in_channel*4)
        if self.input_number >= 3:
            self.linear_c3 = MLP(input_dim=in_channel, embed_dim=in_channel*4)
        self.linear_c2 = MLP(input_dim=in_channel, embed_dim=in_channel*4)
        self.linear_c1 = MLP(input_dim=in_channel, embed_dim=in_channel*4)

        self.linear_fuse = ConvModule(
            in_channels=in_channel*input_number,
            out_channels=mid_channel,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.res_block = Res_block(mid_channel)
        self.conv_final = conv_block(mid_channel, 48, 3, 1, 1, 'c')
        #self.dropout = nn.Dropout2d(0.1)
       
        self.ps = nn.PixelShuffle(4)
    
    def forward(self, x, s):
        c1, c2, c3, c4, _ = x
        #print(c1.shape, c4.shape)
        n = c4.shape[0]
        if self.input_number >= 4:
            _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2]*2, c4.shape[3]*2)
            _c4 = F.interpolate(_c4, size=(c1.shape[2]*2, c1.shape[3]*2),mode='bilinear',align_corners=False)

        if self.input_number >= 3:
            _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2]*2, c3.shape[3]*2)
            _c3 = F.interpolate(_c3, size=(c1.shape[2]*2, c1.shape[3]*2),mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2]*2, c2.shape[3]*2)
        _c2 = F.interpolate(_c2, size=(c1.shape[2]*2, c1.shape[3]*2),mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2]*2, c1.shape[3]*2)
        
        # Linear Fusion
        if self.input_number == 4:
            x = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4], dim=1))
        elif self.input_number == 3:
            x = self.linear_fuse(torch.cat([_c1, _c2, _c3], dim=1))
        else:
            x = self.linear_fuse(torch.cat([_c1, _c2], dim=1))

        # ARRD decoder
        res = F.interpolate(x, scale_factor=s, mode=self.mode, align_corners=False) 
        x = self.res_block(x)
        x = F.interpolate(x, scale_factor=s, mode=self.mode, align_corners=False)
        x += res

        x_final = self.conv_final(x)
        x_final = self.ps(x_final)
        
        return x_final

# For FPN-like structure
@BACKBONES.register_module()
class ARRD_Multi_Level_Type2(BaseModule):
    def __init__(self, 
        in_channel=256,
        mid_channel=128,
        #embedding_dim=1024, 
        mode='bilinear',
        input_number=4, # Number of level features use for reconstruction
        init_cfg=dict(type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(ARRD_Multi_Level_Type2, self).__init__(init_cfg)
        '''
        mode (str) – algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' 
        | 'bicubic' | 'trilinear' | 'area'. Default: 'bilinear'
        '''
        self.mode = mode 
        self.input_number = input_number
        ## Level of features to use
        if self.input_number >= 4:
            #self.linear_c4 = MLP(input_dim=in_channel, embed_dim=embedding_dim)
            self.linear_c4 = MLP(input_dim=in_channel, embed_dim=in_channel)
        if self.input_number >= 3:
            self.linear_c3 = MLP(input_dim=in_channel, embed_dim=in_channel)
        self.linear_c2 = MLP(input_dim=in_channel, embed_dim=in_channel)
        self.linear_c1 = MLP(input_dim=in_channel, embed_dim=in_channel)

        self.linear_fuse = ConvModule(
            in_channels=in_channel*input_number,
            out_channels=mid_channel,
            kernel_size=1
            #norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.res_block = Res_block(mid_channel)
        self.conv_final = conv_block(mid_channel, 48, 3, 1, 1, 'c')
        self.dropout = nn.Dropout2d(0.1)
       
        self.ps = nn.PixelShuffle(4)
    
    def forward(self, x, s):
        c1, c2, c3, c4, _ = x
        #print(c1.shape, c4.shape)
        n = c4.shape[0]
        if self.input_number >= 4:
            _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
            _c4 = F.interpolate(_c4, size=(c1.shape[2], c1.shape[3]),mode='bilinear',align_corners=False)

        if self.input_number >= 3:
            _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
            _c3 = F.interpolate(_c3, size=(c1.shape[2], c1.shape[3]),mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=(c1.shape[2], c1.shape[3]),mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        
        # Linear Fusion
        if self.input_number == 4:
            x = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4], dim=1))
        elif self.input_number == 3:
            x = self.linear_fuse(torch.cat([_c1, _c2, _c3], dim=1))
        else:
            x = self.linear_fuse(torch.cat([_c1, _c2], dim=1))

        # ARRD decoder
        res = F.interpolate(x, scale_factor=s, mode=self.mode, align_corners=False) 
        x = self.res_block(x)
        x = F.interpolate(x, scale_factor=s, mode=self.mode, align_corners=False)
        x += res

        x = self.dropout(x)
        x_final = self.conv_final(x)
        x_final = self.ps(x_final)
        
        return x_final


# For FPN-like structure
@BACKBONES.register_module()
class ARRD_Multi_Level_Small(BaseModule):
    def __init__(self, 
        in_channel=256,
        mid_channel=128,
        #embedding_dim=1024, 
        mode='bilinear',
        input_number=4, # Number of level features use for reconstruction
        init_cfg=dict(type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(ARRD_Multi_Level_Small, self).__init__(init_cfg)
        '''
        mode (str) – algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' 
        | 'bicubic' | 'trilinear' | 'area'. Default: 'bilinear'
        '''
        self.mode = mode 
        self.input_number = input_number
        ## Level of features to use
        if self.input_number >= 4:
            #self.linear_c4 = MLP(input_dim=in_channel, embed_dim=embedding_dim)
            self.linear_c4 = MLP(input_dim=in_channel, embed_dim=in_channel)
        if self.input_number >= 3:
            self.linear_c3 = MLP(input_dim=in_channel, embed_dim=in_channel)
        self.linear_c2 = MLP(input_dim=in_channel, embed_dim=in_channel)
        self.linear_c1 = MLP(input_dim=in_channel, embed_dim=in_channel)

        self.linear_fuse = ConvModule(
            in_channels=in_channel*input_number,
            out_channels=mid_channel,
            kernel_size=1
            #norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.res_block = Res_block(mid_channel)
        self.conv_final = nn.Sequential(conv_block(mid_channel, 48, 3, 1, 1, 'ca'), 
                                        conv_block(48, 3, 3, 1, 1, 'c'), nn.ReLU())

    
    def forward(self, x, s):
        c1, c2, c3, c4, _ = x
        #print(c1.shape, c4.shape)
        n = c4.shape[0]
        if self.input_number >= 4:
            _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
            _c4 = F.interpolate(_c4, size=(c1.shape[2], c1.shape[3]),mode='bilinear',align_corners=False)
        if self.input_number >= 3:
            _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
            _c3 = F.interpolate(_c3, size=(c1.shape[2], c1.shape[3]),mode='bilinear',align_corners=False)
        if self.input_number >= 2:
            _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
            _c2 = F.interpolate(_c2, size=(c1.shape[2], c1.shape[3]),mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        
        # Linear Fusion (different level)
        if self.input_number == 4:
            x = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4], dim=1))
        elif self.input_number == 3:
            x = self.linear_fuse(torch.cat([_c1, _c2, _c3], dim=1))
        elif self.input_number == 2:
            x = self.linear_fuse(torch.cat([_c1, _c2], dim=1))
        else:
            x = _c1

        # ARRD decoder
        res = F.interpolate(x, scale_factor=s, mode=self.mode, align_corners=False) 
        x = self.res_block(x)
        x = F.interpolate(x, scale_factor=s, mode=self.mode, align_corners=False)
        x += res

        x_final = self.conv_final(x)
        
        return x_final


if __name__ == "__main__":
    SR_decoder = ARRD_Multi_Level(in_channel=256)
    x = [torch.ones([1, 256, 32, 32]), 
        torch.ones([1, 256, 16, 16]), 
        torch.ones([1, 256, 8, 8]),
        torch.ones([1, 256, 4, 4])]
    
    mlp = MLP()
    x_out = SR_decoder(x,2)
    # x_out = SR_decoder(x, s=1.6)
    # print(x_out.shape)
    #print()
