import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFusion(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n))

    def forward(self, inputs):
        w = F.relu(self.w)
        w = w / (w.sum() + 1e-6)
        out = 0
        for i, x in enumerate(inputs):
            out += w[i] * x
        return out


class AGBiFPN(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.fuse2 = WeightedFusion(2)
        self.fuse3 = WeightedFusion(3)

        self.conv = lambda: nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU()
        )

        self.td_conv = self.conv()
        self.bu_conv = self.conv()

    def forward(self, p3, p4, p5):
        # Top-down
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode="nearest")
        p4_td = self.td_conv(self.fuse2([p4, p5_up]))

        p4_up = F.interpolate(p4_td, size=p3.shape[2:], mode="nearest")
        p3_td = self.td_conv(self.fuse2([p3, p4_up]))

        # Bottom-up
        p3_down = F.max_pool2d(p3_td, 2)
        p4_out = self.bu_conv(self.fuse3([p4_td, p3_down, p5]))

        return p3_td, p4_out, p5
