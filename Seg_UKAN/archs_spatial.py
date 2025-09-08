# architecture.py
import math
import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_

# import KANLinear, KAN from your kan.py
# from kan import KANLinear, KAN
# If you don't have KANLinear available or want pure linear fallback, pass no_kan=True

from spatial import SpatialAttention

__all__ = ['KANLayerSlim', 'KANBlock', 'PatchEmbed', 'ConvLayer', 'D_ConvLayer', 'UKAN_Light']

# ---------------------------
# Helper modules (DW conv + BN + ReLU)
# ---------------------------
class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, H, W):
        # x: (B, N, C)
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

# ---------------------------
# Slim KANLayer: 2 fc + 2 dwconv
# ---------------------------
class KANLayerSlim(nn.Module):
    """
    Slimmed KANLayer: fc1 -> dwconv1 -> fc2 -> dwconv2
    If no_kan=True, uses nn.Linear instead of KANLinear.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.no_kan = no_kan

        # Note: try to use your KANLinear if available. Otherwise fallback to nn.Linear
        if not no_kan:
            try:
                from kan import KANLinear
                self.fc1 = KANLinear(in_features, hidden_features)
                self.fc2 = KANLinear(hidden_features, out_features)
            except Exception:
                # fallback
                self.fc1 = nn.Linear(in_features, hidden_features)
                self.fc2 = nn.Linear(hidden_features, out_features)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

        self.dw1 = DW_bn_relu(hidden_features)
        self.dw2 = DW_bn_relu(out_features)

        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
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
        # x: (B, N, C)
        B, N, C = x.shape

        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, -1).contiguous()
        x = self.dw1(x, H, W)

        x = self.fc2(x.reshape(B * N, x.shape[-1]))
        x = x.reshape(B, N, -1).contiguous()
        x = self.dw2(x, H, W)

        return x

# ---------------------------
# KAN Block wrapper (LayerNorm + residual)
# ---------------------------
class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        self.layer = KANLayerSlim(in_features=dim, hidden_features=dim, out_features=dim, drop=drop, no_kan=no_kan)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
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
        # x: (B, N, C)
        x = x + self.drop_path(self.layer(self.norm(x), H, W))
        return x

# ---------------------------
# PatchEmbed (same as before)
# ---------------------------
class PatchEmbed(nn.Module):
    """ Image -> patch embeddings (conv-based) """
    def __init__(self, img_size=224, patch_size=3, stride=2, in_chans=3, embed_dim=64):
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
            if m.bias is not None:
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
        x = x.flatten(2).transpose(1, 2)   # (B, N, C)
        x = self.norm(x)
        return x, H, W

# ---------------------------
# Simple conv encoder block (two convs)
# ---------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ---------------------------
# Decoder conv block (same pattern, slightly different channels)
# ---------------------------
class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# ---------------------------
# UKAN Light: 3 encoders, 1 bottleneck, 3 decoders (spatial attention only)
# ---------------------------
class UKAN(nn.Module):
    def __init__(self, num_classes, input_channels=1, img_size=224,
                 embed_dims=[64, 128, 256], no_kan=False, drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
        """
        embed_dims: [e1, e2, e3] -> e1=64, e2=128, e3=256 by default
        """
        super().__init__()
        assert len(embed_dims) == 3, "embed_dims must be length 3"

        e1, e2, e3 = embed_dims

        # small conv encoders (could be removed if you prefer pure patch embed pipeline)
        self.encoder1 = ConvLayer(input_channels, e1)
        self.encoder2 = ConvLayer(e1, e2)
        self.encoder3 = ConvLayer(e2, e3)

        # patch embeds convert conv feature maps -> tokens for KANBlocks
        # here img_size is full image size; we set patch sizes/strides to downsample similarly to prior design
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=3, stride=2, in_chans=e1, embed_dim=e1)
        self.patch_embed2 = PatchEmbed(img_size=img_size // 2, patch_size=3, stride=2, in_chans=e2, embed_dim=e2)
        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=e3, embed_dim=e3)

        # KAN blocks on each encoder stage (light)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 3)]
        self.block1 = nn.ModuleList([KANBlock(dim=e1, drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer, no_kan=no_kan)])
        self.block2 = nn.ModuleList([KANBlock(dim=e2, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer, no_kan=no_kan)])
        self.block3 = nn.ModuleList([KANBlock(dim=e3, drop=drop_rate, drop_path=dpr[2], norm_layer=norm_layer, no_kan=no_kan)])

        # Bottleneck KAN (256 channels)
        self.bottleneck = KANBlock(dim=e3, drop=drop_rate, drop_path=0., norm_layer=norm_layer, no_kan=no_kan)

        # Decoder convs
        self.decoder1 = D_ConvLayer(e3, e2)      # up -> concat with encoder3 (e3 conv output + e3->e2 conv result)
        self.decoder2 = D_ConvLayer(e2, e1)
        self.decoder3 = D_ConvLayer(e1, e1)

        # small 1x1 adapters to reduce concat channels after skip (we removed ChannelLinear)
        # after concat we run SpatialAttention then adapter conv to set to decoder channel
        # 1x1 adapters after skip concat (decoder_out + encoder_out)
        self.adapt1 = nn.Conv2d(e2 + e3, e2, kernel_size=1)  # decoder1 (e2) + t3 (e3)
        self.adapt2 = nn.Conv2d(e1 + e2, e1, kernel_size=1)  # decoder2 (e1) + t2 (e2)
        self.adapt3 = nn.Conv2d(e1 + e1, e1, kernel_size=1)  # decoder3 (e1) + t1 (e1)


        # spatial attention on skips
        self.spatial1 = SpatialAttention(kernel_size=7)
        self.spatial2 = SpatialAttention(kernel_size=7)
        self.spatial3 = SpatialAttention(kernel_size=7)

        # final head
        self.final = nn.Conv2d(e1, num_classes, kernel_size=1)

        # initialize
        self._init_weights()

    def _init_weights(self):
        # initialize convs
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= max(1, m.groups)
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]

        # --- encoder conv stack (simple convs to produce features for skip connections) ---
        e1 = self.encoder1(x)   # (B, e1, H/1, W/1)
        e2 = self.encoder2(F.max_pool2d(e1, 2))  # downsample once -> (B, e2, H/2, W/2)
        e3 = self.encoder3(F.max_pool2d(e2, 2))  # (B, e3, H/4, W/4)

        # --- Tokenize & KAN blocks for each encoder stage ---
        # Stage 1 tokens
        t1_tokens, H1, W1 = self.patch_embed1(e1)
        for blk in self.block1:
            t1_tokens = blk(t1_tokens, H1, W1)
        t1 = t1_tokens.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()  # (B, e1, H1, W1)

        # Stage 2 tokens
        t2_tokens, H2, W2 = self.patch_embed2(e2)
        for blk in self.block2:
            t2_tokens = blk(t2_tokens, H2, W2)
        t2 = t2_tokens.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()  # (B, e2, H2, W2)

        # Stage 3 tokens
        t3_tokens, H3, W3 = self.patch_embed3(e3)
        for blk in self.block3:
            t3_tokens = blk(t3_tokens, H3, W3)
        t3 = t3_tokens.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()  # (B, e3, H3, W3)

        # Bottleneck: process t3 tokens with bottleneck KANBlock
        b_tokens = t3.flatten(2).transpose(1, 2)   # (B, N, C)
        b_tokens = self.bottleneck(b_tokens, H3, W3)
        b = b_tokens.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()  # (B, e3, H3, W3)

        # --- Decoder stage 1: b -> up -> concat with t3 (skip) ---
        x = F.relu(F.interpolate(self.decoder1(b), scale_factor=2, mode='bilinear', align_corners=False))  # up to size of t3*2
        # but ensure t3 spatial dims match: t3 has (H3, W3) while x now is up by 2 -> we'll interpolate to match
        # align sizes: upsample x to same spatial as t3 (we want to concatenate with t3 which is same resolution as b)
        # in this design decoder1 produced e2-sized features, so we want to concat with t3 (encoder stage 3). For safety we align:
        if x.shape[2:] != t3.shape[2:]:
            x = F.interpolate(x, size=t3.shape[2:], mode='bilinear', align_corners=False)
        cat1 = torch.cat([x, t3], dim=1)   # (B, e3 + e3, H3, W3)
        att1 = self.spatial1(cat1)         # (B,1,H3,W3)
        cat1 = cat1 * att1                 # apply spatial attention
        cat1 = self.adapt1(cat1)           # reduce to e2 channels

        # --- Decoder stage 2: up -> concat with t2 ---
        x = F.relu(F.interpolate(self.decoder2(cat1), scale_factor=2, mode='bilinear', align_corners=False))
        if x.shape[2:] != t2.shape[2:]:
            x = F.interpolate(x, size=t2.shape[2:], mode='bilinear', align_corners=False)
        cat2 = torch.cat([x, t2], dim=1)   # (B, e2 + e2, H2, W2)
        att2 = self.spatial2(cat2)
        cat2 = cat2 * att2
        cat2 = self.adapt2(cat2)           # reduce to e1 channels

        # --- Decoder stage 3: up -> concat with t1 ---
        x = F.relu(F.interpolate(self.decoder3(cat2), scale_factor=2, mode='bilinear', align_corners=False))
        if x.shape[2:] != t1.shape[2:]:
            x = F.interpolate(x, size=t1.shape[2:], mode='bilinear', align_corners=False)
        cat3 = torch.cat([x, t1], dim=1)   # (B, e1 + e1, H1, W1)
        att3 = self.spatial3(cat3)
        cat3 = cat3 * att3
        cat3 = self.adapt3(cat3)           # now final decoder feature (B, e1, H1, W1)

        # final upsample to original input size if needed
        out = F.interpolate(cat3, scale_factor=1, mode='bilinear', align_corners=False)
        out = self.final(out)              # (B, num_classes, H, W)
        return out
