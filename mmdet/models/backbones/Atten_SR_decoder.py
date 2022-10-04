import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
# from torch.nn.functional import interpolate, conv2d
from ..builder import BACKBONES
# from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
#                       kaiming_init)
# from mmcv.runner import BaseModule

'''
Attention_Decoder for SR and Degradation Estimation.
By Cui Ziteng, cui@mi.t.u-tokyo.ac.jp
'''

class query_Attention(nn.Module):
    def __init__(self, dim, q_number=22, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        # q_number: contronl the learned query number: 1(resolution) + 1(noise) + K (kernel_dimension)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5
        self.q_number = q_number
        self.hp = nn.Parameter(torch.ones((1, self.q_number, dim)), requires_grad=True)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape   # B:batch, N:h*w, C:channel
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q(self.hp).expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #print('000', q.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, self.q_number, C)
        x = self.proj(x)
        #print('111', x.shape)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class query_SABlock(nn.Module):
    def __init__(self, dim, q_number=22, num_heads=2, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=1e-4):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = query_Attention(
            dim,
            q_number,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((1, 1, dim)),requires_grad=True)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma * self.mlp(self.norm2(x)))
        return x  


# Blind kernel, resolution, noise estimation
@BACKBONES.register_module()
class Deg_pred_ablition(nn.Module):
    def __init__(self, in_channels=64, q_number=19, num_heads=4):
        super(Deg_pred_ablition, self).__init__()
        self.r_base = nn.Parameter(0.5*torch.ones((2)), requires_grad=False)  # basic resolution value
        self.n_base = nn.Parameter(0.5*torch.ones((2)), requires_grad=False)  # basic noise matrix
        # main blocks
        self.generator = query_SABlock(dim=in_channels, q_number=q_number, num_heads=num_heads)
        self.r_linear = nn.Linear(in_channels, 1)
        self.n_linear = nn.Linear(in_channels, 1)
        self.k_linear = nn.Linear(in_channels, 1)

        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if name == 'generator.attn.v.weight':
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        
        x = self.generator(x)
        resolution, noise, kernel = x[:, 0:2], x[:, 2:4], x[:, 4:]
        #print('1111',resolution.shape, noise.shape, kernel.shape)

        r_preds = self.r_linear(resolution).squeeze(-1) + self.r_base
        n_preds = self.n_linear(noise).squeeze(-1) + self.n_base
        #print('000', kernel.shape)
        k_preds = self.k_linear(kernel).squeeze(-1)
        #print('111', k_preds.shape)
        # Return the predicted parameters
        #print('2222',r_preds.shape, n_preds.shape, k_preds.shape)
        rnk_preds = torch.cat([r_preds, n_preds, k_preds], dim=-1)
        #return r_preds, n_preds, k_preds
        #print(rnk_preds.shape)
        return rnk_preds

# Blind kernel, resolution, noise estimation
@BACKBONES.register_module()
class Deg_pred(nn.Module):
    def __init__(self, in_channels=64, q_number=22, num_heads=4):
        super(Deg_pred, self).__init__()
        self.r_base = nn.Parameter(0.5*torch.ones((1)), requires_grad=False)  # basic resolution value
        self.n_base = nn.Parameter(0.5*torch.ones((1)), requires_grad=False)  # basic noise matrix
        # main blocks
        self.generator = query_SABlock(dim=in_channels, q_number=q_number, num_heads=num_heads)
        self.r_linear = nn.Linear(in_channels, 1)
        self.n_linear = nn.Linear(in_channels, 1)
        self.k_linear = nn.Linear(in_channels, 1)

        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if name == 'generator.attn.v.weight':
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        
        x = self.generator(x)
        resolution, noise, kernel = x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1), x[:, 2:]
        r_preds = self.r_linear(resolution).squeeze(-1) + self.r_base
        n_preds = self.n_linear(noise).squeeze(-1) + self.n_base
        #print('000', kernel.shape)
        k_preds = self.k_linear(kernel).squeeze(-1)
        rnk_preds = torch.cat([r_preds, n_preds, k_preds], dim=-1)
        #return r_preds, n_preds, k_preds
        return rnk_preds



# Predict noise and blur kernel
@BACKBONES.register_module()
class Noise_kernel_pred(nn.Module):
    def __init__(self, in_channels=64, q_number=1, num_heads=4):
        super(Noise_kernel_pred, self).__init__()
        self.n_base = nn.Parameter(0.5*torch.ones((1)), requires_grad=True)  # basic noise matrix
        # main blocks
        self.generator = query_SABlock(dim=in_channels, q_number=q_number, num_heads=num_heads)
        self.n_linear = nn.Linear(in_channels, 1)
        self.k_linear = nn.Linear(in_channels, 1)

        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if name == 'generator.attn.v.weight':
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        x = self.generator(x)
        noise, kernel = x[:, 0].unsqueeze(1), x[:, 1:]

        n_preds = self.n_linear(noise).squeeze(-1) + self.n_base
        k_preds = self.k_linear(kernel).squeeze(-1)
        nk_preds = torch.cat([n_preds, k_preds], dim=-1)
        return nk_preds


# Only learn resolution
@BACKBONES.register_module()
class Resolution_pred(nn.Module):
    def __init__(self, in_channels=64, q_number=1, num_heads=4):
        super(Resolution_pred, self).__init__()
        self.r_base = nn.Parameter(0.5*torch.ones((1)), requires_grad=True)  # basic resolution value
        # main blocks
        self.generator = query_SABlock(dim=in_channels, q_number=q_number, num_heads=num_heads)
        self.r_linear = nn.Linear(in_channels, 1)

        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if name == 'generator.attn.v.weight':
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        
        x = self.generator(x)
        resolution = x[:, 0].unsqueeze(1)
        r_preds = self.r_linear(resolution).squeeze(-1) + self.r_base
        return r_preds
        


# Initial Convolution, down 2 resolution
def conv_embedding(in_channels, out_channels):
    proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.GELU(),
            nn.Conv2d(out_channels , out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
    return proj

# Residual Block
class Res_block(nn.Module):
    def __init__(self, in_c=48, mid_c=16):
        super(Res_block, self).__init__()
        self.conv_init = nn.Sequential(nn.Conv2d(in_c, mid_c, 3, 1, 1, bias=True), nn.GELU())

        self.conv_block = nn.Sequential(nn.Conv2d(mid_c, mid_c//2, 3, 1, 1, bias=True), nn.GELU(),
                                        nn.Conv2d(mid_c//2, mid_c, 3, 1, 1, bias=True))

    def forward(self, x):
        x = self.conv_init(x)
        add = x
        x = self.conv_block(x)
        x = x + add
        return x


@BACKBONES.register_module()
class Atten_SR_Decoder(nn.Module):
    def __init__(self, q_number=22, out_size=(128, 128), in_c=64, out_c=3, ps_rate=4):
        super(Atten_SR_Decoder, self).__init__()

        # channel number before pixel shuffle layer
        self.q_number = q_number
        self.out_size = out_size
        mid_c = out_c*ps_rate*ps_rate   #48
        #self.conv_embedding = conv_embedding(in_c, in_c)
        self.para_pred = Deg_pred(in_c, q_number)
        self.Res1 = Res_block(in_c+q_number, mid_c)
        #self.Res2 = Res_block(mid_c, mid_c)
        self.ps = nn.PixelShuffle(ps_rate)    # Up x4

    def forward(self,x):
        #x_pred = self.conv_embedding(x) 
        preds = self.para_pred(x)
        preds_un = preds.unsqueeze(2).unsqueeze(3)
        P = torch.ones([x.shape[0],self.q_number,x.shape[2],x.shape[3]]).to(torch.device(x.device))
        x = torch.cat([x,preds_un*P], dim=1)
        x = self.Res1(x)
        x = F.interpolate(x, self.out_size, mode='bilinear', align_corners=True)
        #x = self.Res2(self.Res1(x))
        x = self.ps(x)
        return preds, x
