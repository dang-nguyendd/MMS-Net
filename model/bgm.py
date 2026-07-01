import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryGuidedModule(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.bgm = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.bgm(x)
