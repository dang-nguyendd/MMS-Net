import torch
import torch.nn as nn

class PathBlockF(nn.Module):
    """
    Path Block F
    F_scale = 1
    Stride = 1
    DF = 1
    256x256 -> 128×128
    """
    # TODO

    def __init__(self, in_ch):
        super().__init__()


        self.path_block_e = nn.Sequential(

            nn.Conv2d(in_ch, in_ch*2, 3, padding=1),
            nn.BatchNorm2d(in_ch*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch*2, in_ch*2, 3, padding=1),
            nn.BatchNorm2d(in_ch*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch*2, in_ch*2, 3, padding=1),
            nn.BatchNorm2d(in_ch*2),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(2, 2),

            nn.Conv2d(in_ch*2, in_ch*2, 3, padding=1),
            nn.BatchNorm2d(in_ch*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch*2, in_ch*2, 3, padding=1),
            nn.BatchNorm2d(in_ch*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch*2, in_ch*2, 3, padding=1),
            nn.BatchNorm2d(in_ch*2),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        x = self.path_block_e(x)

        return x
