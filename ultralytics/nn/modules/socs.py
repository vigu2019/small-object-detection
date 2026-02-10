import torch
import torch.nn as nn

class SOCS(nn.Module):
    """
    Small Object Context Strengthening (SOCS)
    """
    def __init__(self, ch):
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU()
        )

    def forward(self, x):
        return x + self.context(x)
