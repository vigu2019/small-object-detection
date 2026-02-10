import torch
import torch.nn as nn

class DecoupledHead(nn.Module):
    """
    Decoupled Detection Head for YOLOv8
    Separates classification and regression tasks
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Classification branch
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, num_classes, 1)
        )

        # Regression branch
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, 4, 1)
        )

    def forward(self, x):
        cls_out = self.cls_head(x)
        reg_out = self.reg_head(x)
        return cls_out, reg_out

