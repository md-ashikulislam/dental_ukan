import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_
from kan import KANLinear, KAN

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block with channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ConvLayer_SE(nn.Module):
    """ConvLayer with integrated SE block"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_ch)

    def forward(self, x):
        return self.se(self.conv(x))

class D_ConvLayer_SE(nn.Module):
    """Decoder ConvLayer with SE"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_ch)

    def forward(self, x):
        return self.se(self.conv(x))

class DW_bn_relu_SE(nn.Module):
    """Depthwise Conv with SE"""
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()
        self.se = SEBlock(dim, reduction)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.relu(self.bn(self.dwconv(x)))
        x = self.se(x)
        return x.flatten(2).transpose(1, 2)

class KANLayer_SE(nn.Module):
    """KAN Layer with SE-enhanced DWConv"""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        if not no_kan:
            self.fc1 = KANLinear(in_features, hidden_features)
            self.fc2 = KANLinear(hidden_features, out_features)
            self.fc3 = KANLinear(hidden_features, out_features)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        self.dwconv1 = DW_bn_relu_SE(hidden_features)
        self.dwconv2 = DW_bn_relu_SE(hidden_features)
        self.dwconv3 = DW_bn_relu_SE(hidden_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        
        x = self.fc1(x.reshape(B*N, C)).reshape(B, N, C)
        x = self.dwconv1(x, H, W)
        
        x = self.fc2(x.reshape(B*N, C)).reshape(B, N, C)
        x = self.dwconv2(x, H, W)
        
        x = self.fc3(x.reshape(B*N, C)).reshape(B, N, C)
        x = self.dwconv3(x, H, W)
        
        return self.drop(x)

class KANBlock_SE(nn.Module):
    """KAN Block with Squeeze-and-Excitation"""
    def __init__(self, dim, drop=0., drop_path=0., norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        self.norm = norm_layer(dim)
        self.kan_layer = KANLayer_SE(
            in_features=dim, 
            hidden_features=dim,
            drop=drop,
            no_kan=no_kan
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x, H, W):
        x = x + self.drop_path(self.kan_layer(self.norm(x), H, W))
        return x

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                            padding=(patch_size[0]//2, patch_size[1]//2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class UKAN_SE(nn.Module):
    """Complete UKAN-SE Architecture"""
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                img_size=224, embed_dims=[256, 320, 512], no_kan=False, **kwargs):
        super().__init__()
        
        # Encoder with SE
        self.encoder1 = ConvLayer_SE(input_channels, embed_dims[0]//8)
        self.encoder2 = ConvLayer_SE(embed_dims[0]//8, embed_dims[0]//4)
        self.encoder3 = ConvLayer_SE(embed_dims[0]//4, embed_dims[0])
        
        # KAN-SE Blocks
        self.patch_embed3 = PatchEmbed(img_size//4, 3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size//8, 3, 2, embed_dims[1], embed_dims[2])
        
        self.block1 = nn.ModuleList([KANBlock_SE(embed_dims[1], no_kan=no_kan)])
        self.block2 = nn.ModuleList([KANBlock_SE(embed_dims[2], no_kan=no_kan)])
        
        # Decoder with SE
        self.decoder1 = D_ConvLayer_SE(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer_SE(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer_SE(embed_dims[0], embed_dims[0]//4)
        self.decoder4 = D_ConvLayer_SE(embed_dims[0]//4, embed_dims[0]//8)
        self.decoder5 = D_ConvLayer_SE(embed_dims[0]//8, embed_dims[0]//8)
        
        self.norm3 = nn.LayerNorm(embed_dims[1])
        self.norm4 = nn.LayerNorm(embed_dims[2])
        self.dnorm3 = nn.LayerNorm(embed_dims[1])
        self.dnorm4 = nn.LayerNorm(embed_dims[0])
        
        self.final = nn.Conv2d(embed_dims[0]//8, num_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        
        # Encoder
        t1 = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t2 = F.relu(F.max_pool2d(self.encoder2(t1), 2, 2))
        t3 = F.relu(F.max_pool2d(self.encoder3(t2), 2, 2))
        
        # KAN-SE Blocks
        out, H, W = self.patch_embed3(t3)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out
        
        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        # Decoder
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2,2), mode='bilinear'))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2,2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2,2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2,2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2,2), mode='bilinear'))
        
        return self.final(out)