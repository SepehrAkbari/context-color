import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(in_ch, out_ch)
        )
    def forward(self, x):
        return self.net(x)

class ColorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 64, kernel=5, stride=2, padding=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 512, stride=2),
        )
        self.decoder = nn.Sequential(
            UpBlock(512, 256),
            UpBlock(256, 128),
            UpBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, L):
        feats = self.encoder(L)
        ab = self.decoder(feats)
        return ab
