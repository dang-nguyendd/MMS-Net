import torch
import torch.nn as nn

class PathBlockE(nn.Module):
    """
    Path Block E
    F_scale = 2
    Stride = 2
    DF = 2
    256x256 -> 128×128
    """

    def __init__(self, in_ch, dilation=2, stride=2):
        super().__init__()


        self.path_block_e = nn.Sequential(

            nn.Conv2d(in_ch, in_ch*2, 3, padding=dilation, dilation=dilation, stride=stride),
            nn.BatchNorm2d(in_ch*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch*2, in_ch*2, 3, padding=1),
            nn.BatchNorm2d(in_ch*2),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(2, 2),

            nn.Conv2d(in_ch*2, in_ch*4, 3, padding=1),
            nn.BatchNorm2d(in_ch*4),
            nn.ReLU(inplace=True),

            # nn.Conv2d(in_ch*4, in_ch*4, 3, padding=1),
            # nn.BatchNorm2d(in_ch*4),
            # nn.ReLU(inplace=True),

            nn.Conv2d(in_ch*4, in_ch*4, 3, padding=1),
            nn.BatchNorm2d(in_ch*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_ch*4, in_ch*2, 2, stride=2),
            nn.BatchNorm2d(in_ch*2),
            nn.ReLU(inplace=True),

        )


    def forward(self, x):
        x = self.path_block_e(x)

        return x
