import torch
from torch import nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math
from kan import KANLinear, KAN

_all_ = ['AttUKAN'] 

# Attention Block from Attention U-Net
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        
        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        if not no_kan:
            self.fc1 = KANLinear(
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
            self.fc2 = KANLinear(
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
            self.fc3 = KANLinear(
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
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)
    
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
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B*N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B*N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B*N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)
        return x

class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
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
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))
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

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
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
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class AttUKAN(nn.Module):
    def __init__(self, num_classes, input_channels=3, img_size=224, embed_dims=[256, 320, 512], **kwargs):
        super().__init__()
        
        # Encoder remains unchanged
        self.encoder1 = ConvLayer(input_channels, embed_dims[0]//8)
        self.encoder2 = ConvLayer(embed_dims[0]//8, embed_dims[0]//4)
        self.encoder3 = ConvLayer(embed_dims[0]//4, embed_dims[0])
        
        # KAN Blocks
        self.block1 = nn.ModuleList([KANBlock(dim=embed_dims[1])])
        self.block2 = nn.ModuleList([KANBlock(dim=embed_dims[2])])
        self.dblock1 = nn.ModuleList([KANBlock(dim=embed_dims[1])])
        self.dblock2 = nn.ModuleList([KANBlock(dim=embed_dims[0])])
        
        # Patch Embeddings
        self.patch_embed3 = PatchEmbed(img_size//4, 3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size//8, 3, 2, embed_dims[1], embed_dims[2])
        
        # Decoder with Attention-only
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.att1 = Attention_block(F_g=embed_dims[1], F_l=embed_dims[1], F_int=embed_dims[1]//2)
        
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.att2 = Attention_block(F_g=embed_dims[0], F_l=embed_dims[0], F_int=embed_dims[0]//2)
        
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0]//4)
        self.att3 = Attention_block(F_g=embed_dims[0]//4, F_l=embed_dims[0]//4, F_int=embed_dims[0]//8)
        
        self.decoder4 = D_ConvLayer(embed_dims[0]//4, embed_dims[0]//8)
        
        self.final = nn.Conv2d(embed_dims[0]//8, num_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        
        # Encoder
        x1 = F.max_pool2d(self.encoder1(x), 2, 2)  # /2
        x2 = F.max_pool2d(self.encoder2(x1), 2, 2)  # /4
        x3 = F.max_pool2d(self.encoder3(x2), 2, 2)  # /8
        
        # KAN Stages
        x4, H4, W4 = self.patch_embed3(x3)
        for blk in self.block1: x4 = blk(x4, H4, W4)
        x4 = x4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2)
        
        x5, H5, W5 = self.patch_embed4(x4)
        for blk in self.block2: x5 = blk(x5, H5, W5)
        x5 = x5.reshape(B, H5, W5, -1).permute(0, 3, 1, 2)
        
        # Decoder with Attention (no fusion)
        d4 = F.interpolate(self.decoder1(x5), scale_factor=2, mode='bilinear')
        d4 = self.att1(g=d4, x=x4)  # Attention only
        
        d3 = F.interpolate(self.decoder2(d4), scale_factor=2, mode='bilinear')
        d3 = self.att2(g=d3, x=x3)  # Attention only
        
        d2 = F.interpolate(self.decoder3(d3), scale_factor=2, mode='bilinear')
        d2 = self.att3(g=d2, x=x2)  # Attention only
        
        d1 = F.interpolate(self.decoder4(d2), scale_factor=2, mode='bilinear')
        
        return self.final(d1)