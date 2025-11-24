import torch
import torch.nn as nn

class FeatureBooster(nn.Module):
    """
    Feature Booster
    F_scale = 1
    Stride = 1
    DF = 1
    Filters = 16 (default)
    """

    def __init__(self, in_ch, out_ch=16):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)

        # 3 repeated conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x
