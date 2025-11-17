import torch
import torch.nn as nn

from path_block_a import PathBlockA
from path_block_b import PathBlockB
from path_block_c import PathBlockC
from path_block_d import PathBlockD
from path_block_e import PathBlockE
from bottleneck import Bottleneck

class PathBlock(nn.Module):
    """
    One of the 3 parallel paths:
    
    """
    def __init__(self, in_ch, mid_ch, out_ch, dilation=2):
        super().__init__()

        self.dilated_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=mid_ch,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )

        self.conv = nn.Conv2d(
            in_channels=mid_ch,
            out_channels=mid_ch,
            kernel_size=3,
            padding=1
        )

        self.bn = nn.BatchNorm2d(mid_ch)
        self.relu = nn.ReLU(inplace=True)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Upsample 2×
        self.tconv = nn.ConvTranspose2d(
            in_channels=mid_ch,
            out_channels=out_ch,
            kernel_size=2,
            stride=2
        )

    def forward(self, x):
        x = self.dilated_conv(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.tconv(x)
        return x


class MultiPathModel(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()

        # Input stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Three parallel paths (with different dilations)
        self.path1 = PathBlock(base_channels, base_channels, base_channels, dilation=2)
        self.path2 = PathBlock(base_channels, base_channels, base_channels, dilation=3)
        self.path3 = PathBlock(base_channels, base_channels, base_channels, dilation=4)

    def forward(self, x):
        x = self.stem(x)

        # Parallel branches
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)

        # Depth-wise (channel dimension) concatenation
        fused = torch.cat([p1, p2, p3], dim=1)

        return fused


# Example
if __name__ == "__main__":
    model = MultiPathModel()
    inp = torch.randn(1, 3, 128, 128)
    out = model(inp)
    print("Output shape:", out.shape)
