import torch
import torch.nn as nn

class PathBlockA(nn.Module):
    """
    Path Block A
    F_scale = 2
    Stride = 2
    DF = 2
    """

    def __init__(self, in_ch, out_ch, dilation=2, stride=2):
        super().__init__()

        # --- stage 1 ---
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # pool 1
        self.pool1 = nn.AvgPool2d(2, 2)

        # --- 3 repeated conv blocks ---
        self.block2 = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
                for _ in range(3)
            ]
        )

        # pool 2
        self.pool2 = nn.AvgPool2d(2, 2)

        # --- 2 repeated conv blocks ---
        self.block3 = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
                for _ in range(2)
            ]
        )

        # deconv
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.deconv(x)

        return x
