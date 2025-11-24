import torch
import torch.nn as nn

class PathBlockB(nn.Module):
    """
    Path Block B
    F_scale = 1
    Stride = 1
    DF = 1
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        # First conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # factory for a single conv-bn-relu block
        def block():
            return nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # three 3-conv stacks
        self.block1 = nn.Sequential(block(), block(), block())
        self.block2 = nn.Sequential(block(), block(), block())
        self.block3 = nn.Sequential(block(), block(), block())

        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)

        x = self.block1(x)
        x = self.pool(x)

        x = self.block2(x)
        x = self.pool(x)

        x = self.block3(x)

        return x
