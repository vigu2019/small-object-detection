import torch
import torch.nn as nn


class ELAM(nn.Module):
    """
    Edge-aware Localization Attention Module (ELAM)
    Refines object boundaries for better localization of small objects.
    """

    def __init__(self, channels):
        super().__init__()

        # Edge enhancement (depthwise conv)
        self.edge_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )

        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        edge_feat = self.edge_conv(x)
        att = self.channel_att(edge_feat)
        out = x + edge_feat * att
        return self.bn(out)
