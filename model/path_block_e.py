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

    def __init__(self, in_ch, out_ch, dilation=2, stride=2):
        super().__init__()

        # Dilated conv
        self.dilated = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AvgPool2d(2, 2)

        # factory block: out_ch → out_ch
        def block():
            return nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.block1 = block()

        # 3 repeated conv blocks
        self.block2 = nn.Sequential(block(), block(), block())

        # deconv
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.dilated(x)
        x = self.block1(x)

        x = self.pool(x)

        x = self.block2(x)

        x = self.deconv(x)

        return x
