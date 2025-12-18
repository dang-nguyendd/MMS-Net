import torch
import torch.nn as nn

class PathBlockC(nn.Module):
    """
    Path Block C
    F_scale = 4
    Stride = 4
    DF = 4
    256x256 -> 64x64
    """

    def __init__(self, in_ch, dilation=4, stride=4):
        super().__init__()

        self.path_block_c = nn.Sequential(
            nn.Conv2d(in_ch, in_ch*4, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(in_ch*4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch*4, in_ch*4, 3, padding=1, stride=2),
            nn.BatchNorm2d(in_ch*4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch*4, in_ch*4, 3, padding=1, stride=2),
            nn.BatchNorm2d(in_ch*4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch*4, in_ch*4, 3, padding=1),
            nn.BatchNorm2d(in_ch*4),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(2, 2),

            nn.Conv2d(in_ch*4, in_ch*8, 3, padding=1),
            nn.BatchNorm2d(in_ch*8),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch*8, in_ch*8, 3, padding=1),
            nn.BatchNorm2d(in_ch*8),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch*8, in_ch*8, 3, padding=1),
            nn.BatchNorm2d(in_ch*8),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_ch*8, in_ch*8, 3, padding=1),
            nn.BatchNorm2d(in_ch*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_ch*8, in_ch*4, 2, stride=2),
            nn.BatchNorm2d(in_ch*4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.path_block_c(x)

        return x
