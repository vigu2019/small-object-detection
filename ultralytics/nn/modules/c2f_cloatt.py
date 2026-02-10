import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv
from .block import C2f


class CloAtt(nn.Module):
    """
    CloAtt = Local + Channel + Spatial attention
    Designed to improve small-object detection.
    """
    def __init__(self, c, reduction=16):
        super().__init__()
        r = max(c // reduction, 4)

        # Local attention: depthwise conv
        self.local = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True)
        )

        # Channel attention (SE)
        self.fc = nn.Sequential(
            nn.Conv2d(c, r, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(r, c, 1, bias=False)
        )

        # Spatial attention (CBAM-like)
        self.spatial = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False
        )

    def forward(self, x):
    # FIX: Make sure all submodules are on the same device as input
        device = x.device
        self.fc.to(device)
        self.local.to(device)
        self.spatial.to(device)

    # Local branch
        l = self.local(x)

    # Channel branch
        w = torch.mean(x, dim=(2, 3), keepdim=True)
        w = torch.sigmoid(self.fc(w))
        c = x * w

    # Spatial branch
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        avg_map = torch.mean(x, dim=1, keepdim=True)
        s_mask = torch.sigmoid(self.spatial(torch.cat([max_map, avg_map], dim=1)))
        s = x * s_mask

    # Combine branches
        return l + c + s



class C2fCloAtt(C2f):
    """
    C2f + CloAtt backbone enhancement block.
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.att = CloAtt(c2)

    def forward(self, x):
        y = super().forward(x)
        return self.att(y)
