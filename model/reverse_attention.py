import torch
import torch.nn as nn
import torch.nn.functional as F

class ReverseAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, prediction):
        """
        features: feature map from encoder/decoder (B, C, H, W)
        prediction: coarse prediction map (B, 1, H, W)
        """
        # Reverse attention map
        ra_map = 1 - torch.sigmoid(prediction)

        # Element-wise multiplication
        ra_features = features * ra_map

        # Refine
        out = self.conv(ra_features)
        return out
