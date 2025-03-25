import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math
from kan import KANLinear

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
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
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                             stride=stride, padding=(patch_size[0]//2, patch_size[1]//2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
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

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)
        self.drop = nn.Dropout(drop)

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
        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, 
                             act_layer=act_layer, drop=drop, no_kan=no_kan)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))
        return x

class ConvLayer(nn.Module):
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

    def forward(self, x):
        return self.conv(x)

class D_ConvLayer(nn.Module):
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

    def forward(self, x):
        return self.conv(x)

class UKAN(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False, 
                 img_size=512, embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.target_size = (img_size, img_size)
        
        # Encoder
        self.encoder1 = ConvLayer(3, embed_dims[0]//8)
        self.encoder2 = ConvLayer(embed_dims[0]//8, embed_dims[0]//4)
        self.encoder3 = ConvLayer(embed_dims[0]//4, embed_dims[0])

        # KAN Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2)]
        self.block1 = nn.ModuleList([KANBlock(embed_dims[1], drop_rate, dpr[0], no_kan=no_kan)])
        self.block2 = nn.ModuleList([KANBlock(embed_dims[2], drop_rate, dpr[1], no_kan=no_kan)])
        
        # Patch Embeddings
        self.patch_embed3 = PatchEmbed(img_size//4, 3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size//8, 3, 2, embed_dims[1], embed_dims[2])

        # Decoder
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0]//4)
        self.decoder4 = D_ConvLayer(embed_dims[0]//4, embed_dims[0]//8)
        self.decoder5 = D_ConvLayer(embed_dims[0]//8, embed_dims[0]//8)
        
        # Output heads
        self.final = nn.Sequential(
            nn.Conv2d(embed_dims[0]//8, num_classes, kernel_size=1),
            nn.Upsample(size=self.target_size, mode='bilinear', align_corners=True)
        )
        
        # Deep supervision heads
        if deep_supervision:
            self.ds1 = nn.Sequential(
                nn.Conv2d(embed_dims[1], num_classes, kernel_size=1),
                nn.Upsample(size=self.target_size, mode='bilinear', align_corners=True)
            )
            self.ds2 = nn.Sequential(
                nn.Conv2d(embed_dims[0], num_classes, kernel_size=1),
                nn.Upsample(size=self.target_size, mode='bilinear', align_corners=True)
            )

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

    def forward(self, x):
        B = x.shape[0]
        
        # Encoder
        e1 = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))  # /2
        e2 = F.relu(F.max_pool2d(self.encoder2(e1), 2, 2))  # /4
        e3 = F.relu(F.max_pool2d(self.encoder3(e2), 2, 2))  # /8

        # Bottleneck
        x, H, W = self.patch_embed3(e3)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = x

        x, H, W = self.patch_embed4(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # Decoder
        x = F.interpolate(self.decoder1(x), scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.add(x, t4)
        
        if self.deep_supervision:
            ds1 = self.ds1(x)

        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.dnorm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(self.decoder2(x), scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.add(x, e3)
        
        if self.deep_supervision:
            ds2 = self.ds2(x)

        x = F.interpolate(self.decoder3(x), scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.add(x, e2)
        x = F.interpolate(self.decoder4(x), scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.add(x, e1)
        x = F.interpolate(self.decoder5(x), scale_factor=2, mode='bilinear', align_corners=True)
        
        final_out = self.final(x)
        
        if self.deep_supervision:
            return [ds1, ds2, final_out]  # All outputs are now [B, C, H, W] with H,W=target_size
        return final_out