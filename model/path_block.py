import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class PathBlock(nn.Module, ABC):
    """
    Abstract interface for a path block.
    """
    def __init__(self, in_ch, mid_ch, out_ch, dilation=2, stride=2):
        super().__init__()

        self.dilated_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=mid_ch,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            stride=stride
        )

        self.conv = nn.Conv2d(
            in_channels=mid_ch,
            out_channels=mid_ch,
            kernel_size=3,
            padding=1
        )

        self.bn = nn.BatchNorm2d(mid_ch)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.de_conv = nn.ConvTranspose2d(
            in_channels=mid_ch,
            out_channels=out_ch,
            kernel_size=2,
            stride=2
        )

    @abstractmethod
    def forward(self, x):
        """
        To be implemented by subclasses.
        """
        pass
