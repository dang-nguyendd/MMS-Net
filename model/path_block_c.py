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

        # Dilated conv
        self.dilated = nn.Sequential(
            nn.Conv2d(in_ch, in_ch*4, 3, padding=dilation, dilation=dilation, stride=stride),
            nn.BatchNorm2d(in_ch*4),
            nn.ReLU(inplace=True),
        )

        # Factory for conv → bn → relu blocks (out_ch → out_ch)
        def block_1():
            return nn.Sequential(
                nn.Conv2d(in_ch*4, in_ch*4, 3, padding=1),
                nn.BatchNorm2d(in_ch*4),
                nn.ReLU(inplace=True),
            )
        def block_2_1():
            return nn.Sequential(
                nn.Conv2d(in_ch*4, in_ch*8, 3, padding=1),
                nn.BatchNorm2d(in_ch*8),
                nn.ReLU(inplace=True),
            )
        def block_2_2():
            return nn.Sequential(
                nn.Conv2d(in_ch*8, in_ch*8, 3, padding=1),
                nn.BatchNorm2d(in_ch*8),
                nn.ReLU(inplace=True),
            )
        # 2 blocks
        self.block1 = nn.Sequential(block_1(),block_1(), block_1())

        self.pool = nn.AvgPool2d(2, 2)

        # 4 blocks
        self.block2 = nn.Sequential(block_2_1(), block_2_2(), block_2_2(), block_2_2())

        # Deconv
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch*8, in_ch*4, 2, stride=2),
            nn.BatchNorm2d(in_ch*4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.dilated(x)

        x = self.block1(x)
        x = self.pool(x)

        x = self.block2(x)

        x = self.deconv(x)
        return x
