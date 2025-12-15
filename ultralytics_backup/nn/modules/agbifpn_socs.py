import torch.nn as nn
from .agbifpn import AGBiFPN
from .socs import SOCS

class AGBiFPN_SOCS(nn.Module):
    """
    AGBiFPN + SOCS Neck
    """
    def __init__(self, ch):
        super().__init__()
        self.neck = AGBiFPN(ch)
        self.socs = SOCS(ch)

    def forward(self, inputs):
        p3, p4, p5 = self.neck(inputs)
        p3 = self.socs(p3)  # strengthen small-object feature
        return [p3, p4, p5]
