import torch
import torch.nn as nn

class PathBlockD(nn.Module):
    """
    Path Block D
    F_scale = 4
    Stride = 4
    DF = 4
    256x256 -> 128×128
    """

    def __init__(self, in_ch, out_ch, dilation=4, stride=4):
        super().__init__()

        # Dilated conv
        self.dilated = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # Normal conv (out_ch → out_ch)
        def block():
            return nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        
        # Two repeated conv blocks
        self.block1 = nn.Sequential(block(), block())

        self.pool = nn.AvgPool2d(2, 2)

        # Two repeated conv blocks
        self.block2 = nn.Sequential(block(), block())

        # Two deconv steps
        self.deconv= nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ),
        )

    def forward(self, x):
        x = self.dilated(x)
        x = self.block1(x)

        x = self.pool(x)

        x = self.block2(x)

        x = self.deconv(x)

        return x
