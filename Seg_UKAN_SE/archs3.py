import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KANLinear, KAN  # Ensure KAN is correctly installed and imported

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
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
    def __init__(self, in_features, out_features):
        super(KANLayer, self).__init__()
        self.kan = KANLinear(in_features, out_features)
    
    def forward(self, x):
        return self.kan(x)

class AttUKAN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(AttUKAN, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoding path
        for feature in features:
            self.encoder.append(ConvBlock(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        
        # Decoding path
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(ConvBlock(feature * 2, feature))
        
        # Attention layers
        self.attentions = nn.ModuleList([
            Attention_block(features[i+1], features[i], features[i]) for i in range(len(features) - 1)
        ])
        
        # KAN layers
        self.kan1 = KANLayer(256 * 256, 128 * 128)
        self.kan2 = KANLayer(128 * 128, 64 * 64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.final_activation = nn.Sigmoid()
    
    def forward(self, x):
        skip_connections = []
        
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            if i // 2 < len(self.attentions):
                x = self.attentions[i // 2](x, skip_connections[i // 2])
            x = torch.cat((x, skip_connections[i // 2]), dim=1)
            x = self.decoder[i + 1](x)
        
        # KAN processing
        x = x.view(x.size(0), -1)  # Flatten spatial dimensions
        x = self.kan1(x)
        x = self.kan2(x)
        x = x.view(x.size(0), 64, 64)  # Reshape back
        
        x = self.final_conv(x.unsqueeze(1))
        x = self.final_activation(x)
        
        return x
