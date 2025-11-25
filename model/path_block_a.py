import torch
import torch.nn as nn

class PathBlockA(nn.Module):
    """
    Path Block A
    F_scale = 2
    Stride = 2
    DF = 2
    256x256 -> 64x64
    """

    def __init__(self, in_ch, dilation=2, stride=2):
        super().__init__()

        # --- stage 1 ---
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch*2, 3, padding=dilation, dilation=dilation, stride=stride),
            nn.BatchNorm2d(in_ch*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch*2, in_ch*2, 3, padding=1),
            nn.BatchNorm2d(in_ch*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch*2, in_ch*2, 3, padding=1),
            nn.BatchNorm2d(in_ch*2),
            nn.ReLU(inplace=True),
        )

        # pool 1
        self.pool1 = nn.AvgPool2d(2, 2)

        # --- 3 repeated conv blocks ---
        self.block2_1 = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch*4, 3, padding=1),
            nn.BatchNorm2d(in_ch*4),
            nn.ReLU(inplace=True),
        )
        self.block2_2 = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(in_ch*4, in_ch*4, 3, padding=1),
                    nn.BatchNorm2d(in_ch*4),
                    nn.ReLU(inplace=True),
                )
                for _ in range(2)
            ]
        )

        # pool 2
        self.pool2 = nn.AvgPool2d(2, 2)

        # --- 2 repeated conv blocks ---
        self.block3_1 = nn.Sequential(
            nn.Conv2d(in_ch*4, in_ch*8, 3, padding=1),
            nn.BatchNorm2d(in_ch*8),
            nn.ReLU(inplace=True),
        )

        self.block3_2 = nn.Sequential(
            nn.Conv2d(in_ch*8, in_ch*8, 3, padding=1),
            nn.BatchNorm2d(in_ch*8),
            nn.ReLU(inplace=True),
        )

        # deconv
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch*8, in_ch*4, 2, stride=2),
            nn.BatchNorm2d(in_ch*4),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.pool2(x)

        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.deconv(x)

        return x
