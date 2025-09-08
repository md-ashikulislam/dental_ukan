# spatial.py
import torch
from torch import nn

class SpatialAttention(nn.Module):
    """
    Simple spatial attention module (CBAM-style).
    Input: (B, C, H, W)
    Output: attention map (B, 1, H, W) (sigmoid)
    Typically used as: out = att(x) * x
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = kernel_size // 2
        # two-channel concat: avg + max
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)     # (B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)   # (B,1,H,W)
        cat = torch.cat([avg_out, max_out], dim=1)       # (B,2,H,W)
        attn = self.conv(cat)                            # (B,1,H,W)
        return self.sigmoid(attn)                        # (B,1,H,W)
