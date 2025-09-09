import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *
from einops import rearrange

import timm
from timm.layers import DropPath, to_2tuple, trunc_normal_

import math


import kannet
from torch.nn import init

__all__ = [
    'KANLayer', 'KANBlock', 'DWConv', 'DW_bn_relu3D', 'DW_bn_relu',
    'PatchEmbed3D', 'ECALayer', 'ConvLayer', 'D_ConvLayer3D', 'PANLayer', 'UKAN'
]

class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]

        if not no_kan:
            self.fc1 = kannet.KANLinear(
                        in_features,
                        hidden_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc2 = kannet.KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc3 = kannet.KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            # # TODO   
            # self.fc4 = KANLinear(
            #             hidden_features,
            #             out_features,
            #             grid_size=grid_size,
            #             spline_order=spline_order,
            #             scale_noise=scale_noise,
            #             scale_base=scale_base,
            #             scale_spline=scale_spline,
            #             base_activation=base_activation,
            #             grid_eps=grid_eps,
            #             grid_range=grid_range,
            #         )   

        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        # TODO
        # self.fc1 = nn.Linear(in_features, hidden_features)

        self.dwconv_1 = DW_bn_relu3D(hidden_features)
        self.dwconv_2 = DW_bn_relu3D(hidden_features)
        self.dwconv_3 = DW_bn_relu3D(hidden_features)

        # # TODO
        # self.dwconv_4 = DW_bn_relu(hidden_features)
    
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    

    def forward(self, x, D, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_1(x, D, H, W)
        x = self.fc2(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_2(x, D, H, W)
        x = self.fc3(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_3(x, D, H, W)

        # # TODO
        # x = x.reshape(B,N,C).contiguous()
        # x = self.dwconv_4(x, H, W)
    
        return x

class KANBlock(nn.Module):
    def __init__(self, dim, drop=0.1, drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, D, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), D, H, W))

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class DW_bn_relu3D(nn.Module):
    def __init__(self, dim):
        super(DW_bn_relu3D, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm3d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def to_3tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x, x)


class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=4, embed_dim=1):
        super().__init__()
        patch_size = to_3tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        x = self.norm(x)

        return x, D, H, W


class ECALayer(nn.Module):
    """Constructs a 3D ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.channel = channel
        self.conv = nn.Conv2d(self.channel, 1, kernel_size=(k_size, 1), padding=((k_size - 1) // 2, 0), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, channel):
        # Global spatial information descriptor
        y = self.avg_pool(x)  # Output shape: [batch_size, channels, 1, 1, 1]

        # Adapting ECA for 3D: apply conv along channel dimension
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class D_ConvLayer3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        x = self.conv(input)
        eca = ECALayer(x.shape[0])
        eca = eca.cuda()
        x = eca(x, x.shape[0])
        return x
    

# 定义PAN结构中的单个路径模块
class PANLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PANLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, lateral):
        x = F.interpolate(x, size=lateral.shape[-3:], mode="trilinear", align_corners=False)  # 上采样至lateral大小
        x = torch.cat([x, lateral], dim=1)  # 沿通道维度连接
        x = self.conv(x)  # 3D卷积
        x = self.bn(x)  # 3D批归一化
        return self.relu(x)


class UKAN(nn.Module):
    def __init__(self, num_classes, embed_dims=[128, 160, 256], drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm, depths=[1, 1, 1]):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.encoder1 = D_ConvLayer3D(4, kan_input_dim//8)
        self.encoder2 = D_ConvLayer3D(kan_input_dim//8, kan_input_dim//4)
        self.encoder3 = D_ConvLayer3D(kan_input_dim//4, kan_input_dim)

        self.pan1 = PANLayer(160, 32)
        self.pan2 = PANLayer(48, 16)
        #self.eca1 = ECALayer(32)
        #self.eca2 = ECALayer(16)


        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0], 
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.patch_embed3 = PatchEmbed3D(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed3D(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer3D(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer3D(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer3D(embed_dims[0], embed_dims[0]//4)
        self.decoder4 = D_ConvLayer3D(embed_dims[0]//4*3, embed_dims[0]//8*3)
        self.decoder5 = D_ConvLayer3D(embed_dims[0]//8*5, embed_dims[0]//8)

        self.final = nn.Conv3d(embed_dims[0] // 8, num_classes, kernel_size=1)
        #self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2))
        t3 = out

        # PAN融合特征
        pan1 = self.pan1(t3, t2)  # 从底层向上，融合enc3和enc4特征
        print("pan1:", pan1.shape)
        #pan1 = self.eca1(pan1)
        #print("pan1:", pan1.shape)
        pan2 = self.pan2(pan1, t1)  # 融合上采样的pan3和enc2特征
        print("pan2:", pan2.shape)
        #pan2 = self.eca2(pan2)
        #print("pan2:", pan2.shape)

        ### Tokenized KAN Stage
        ### Stage 4

        out, D, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, D, H, W)
        out = self.norm3(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        t4 = out

        ### Bottleneck

        out, D, H, W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, D, H, W)
        out = self.norm4(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        ### Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True))

        out = torch.add(out, t4)
        _, _, D, H, W = out.shape
        out = out.reshape(out.size(0), out.size(1), -1).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, D, H, W)

        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True))
        out = torch.add(out,t3)
        _,_,D,H,W = out.shape
        out = out.reshape(out.size(0), out.size(1), -1).transpose(1, 2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, D, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True))
        #print("out:", out.shape)
        #print("t2:", t2.shape)
        out = torch.cat((out,t2), dim=1)
        #print("out:", out.shape)
        out = torch.cat((out,pan1), dim=1)
        #print("out:", out.shape)
        #print(out.shape)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True))
        out = torch.cat((out,t1), dim=1)
        out = torch.cat((out,pan2), dim=1)
        #print("out:", out.shape)
        #print(out.shape)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True))

        return self.final(out)