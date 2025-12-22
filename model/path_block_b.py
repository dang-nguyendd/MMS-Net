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
        self.path_block_b = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),

                nn.AvgPool2d(2, 2),

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

                nn.Conv2d(in_ch*2, in_ch*4, 3, padding=1),
                nn.BatchNorm2d(in_ch*4),
                nn.ReLU(inplace=True),
                
                # nn.Conv2d(in_ch*4, in_ch*4, 3, padding=1),
                # nn.BatchNorm2d(in_ch*4),
                # nn.ReLU(inplace=True),

                nn.Conv2d(in_ch*4, in_ch*4, 3, padding=1),
                nn.BatchNorm2d(in_ch*4),
                nn.ReLU(inplace=True),

            )
     

    def forward(self, x):
        
        x = self.path_block_b(x)

        return x
