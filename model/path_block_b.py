import torch
import torch.nn as nn

class PathBlockB(nn.Module):
    """
    Path Block B
    F_scale = 1
    Stride = 1
    DF = 1
    256x256 -> 64x64
    """

    def __init__(self, in_ch):
        super().__init__()

        # factory for a single conv-bn-relu block
        def block_1():
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
            )
        # factory for a single conv-bn-relu block
        def block_2_1():
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch*2, 3, padding=1),
                nn.BatchNorm2d(in_ch*2),
                nn.ReLU(inplace=True),
            )
        
        def block_2_2():
            return nn.Sequential(
                nn.Conv2d(in_ch*2, in_ch*2, 3, padding=1),
                nn.BatchNorm2d(in_ch*2),
                nn.ReLU(inplace=True),
            )
        
        # factory for a single conv-bn-relu block
        def block_3_1():
            return nn.Sequential(
                nn.Conv2d(in_ch*2, in_ch*4, 3, padding=1),
                nn.BatchNorm2d(in_ch*4),
                nn.ReLU(inplace=True),
            )
        def block_3_2():
            return nn.Sequential(
                nn.Conv2d(in_ch*4, in_ch*4, 3, padding=1),
                nn.BatchNorm2d(in_ch*4),
                nn.ReLU(inplace=True),
            )
        # three 3-conv stacks
        self.block1 = nn.Sequential(block_1(), block_1(), block_1(), block_1())
        self.block2 = nn.Sequential(block_2_1(), block_2_2(), block_2_2())
        self.block3 = nn.Sequential(block_3_1(), block_3_2(), block_3_2())

        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool(x)

        x = self.block2(x)
        x = self.pool(x)

        x = self.block3(x)

        return x
