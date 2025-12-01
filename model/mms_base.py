import torch
import torch.nn as nn
from torchinfo import summary

from .path_block_a import PathBlockA
from .path_block_b import PathBlockB
from .path_block_c import PathBlockC
from .path_block_d import PathBlockD
from .path_block_e import PathBlockE
from .feature_booster import FeatureBooster

class MMSNet(nn.Module):
    """
    Full Path
    
    """
    def __init__(self, in_ch = 16, out_ch =2):
        super().__init__()

        self.path_a = PathBlockA(in_ch=in_ch)
        self.path_b = PathBlockB(in_ch=in_ch)
        self.path_c = PathBlockC(in_ch=in_ch)
        self.path_d = PathBlockD(in_ch=in_ch)
        self.path_e = PathBlockE(in_ch=in_ch)

        self.feature_booster_1 = FeatureBooster(in_ch=3)
        self.feature_booster_2 = FeatureBooster(in_ch=in_ch*2)

        self.bn1 = nn.BatchNorm2d(in_ch*4)
        self.bn2 = nn.BatchNorm2d(in_ch*2)
        self.bn3 = nn.BatchNorm2d(in_ch)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=in_ch,
            kernel_size=3,
            padding=1
        )

        self.conv_1x1 = nn.Conv2d(
            in_channels=in_ch*12,
            out_channels=in_ch*4,
            kernel_size=1,
            padding=0
        )

        self.de_conv_1 = nn.ConvTranspose2d(
            in_channels=in_ch*4,
            out_channels=in_ch*2,
            kernel_size=2,
            stride=2
        )

        self.de_conv_2 = nn.ConvTranspose2d(
            in_channels=in_ch*2,
            out_channels=in_ch,
            kernel_size=2,
            stride=2
        )

        self.de_conv_3 = nn.ConvTranspose2d(
            in_channels=in_ch*6 + 16,
            out_channels=in_ch*2,
            kernel_size=2,
            stride=2
        )

        self.conv_1x1_2 = nn.Conv2d(
            in_channels=in_ch*3,
            out_channels=out_ch,
            kernel_size=1,
            padding=0
        )

        # Input stem
        self.in_stem = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(in_ch),        
            nn.ReLU(inplace=True)
        )

        # Mid stem
        self.mid_stem = nn.Sequential(
            self.de_conv_3,
            nn.BatchNorm2d(in_ch*2),        
            nn.ReLU(inplace=True)  
        )

        # Output stem
        self.out_stem = nn.Sequential(
            self.conv_1x1_2,                 
            nn.BatchNorm2d(out_ch),        
            nn.ReLU(inplace=True),        
            nn.Softmax(dim=1),          
        )

    def forward(self, x):
        fb_1 = self.feature_booster_1.forward(x)
        dense_skip_path_1 = x

        x = self.in_stem(x)

        #__________ Cascaded Path 1 __________
        # Parallel branches
        path_a = self.path_a.forward(x)
        path_b = self.path_b.forward(x)
        path_c = self.path_c.forward(x)

        # Depth-wise (channel dimension) concatenation
        fused_1 = torch.cat([path_a, path_b, path_c], dim=1)

        #__________ BottleNeck __________
        x = self.conv_1x1(fused_1)
        x = self.bn1(x)
        x = self.relu1(x)

        z = self.de_conv_1(x)
        z = self.bn2(z)
        z = self.relu2(z)
    
        x = self.de_conv_2(z)
        x = self.bn3(x)
        x = self.relu3(x)

        #__________ Cascaded Path 2 __________
        # Parallel branches
        path_d = self.path_d.forward(x)
        path_e = self.path_e.forward(x)
        fb_2 = self.feature_booster_2.forward(z)
        dense_skip_path_2 = z

        # Depth-wise (channel dimension) concatenation
        fused_2 = torch.cat([path_d, path_e, dense_skip_path_2, fb_2], dim=1)

        x = self.mid_stem(fused_2)

        #__________ Cascaded Path 3 __________
        # Depth-wise (channel dimension) concatenation
        fused_3 = torch.cat([x, fb_1], dim=1)
        
        x = self.out_stem(fused_3)

        return x



if __name__ == "__main__":
    model = MMSNet()
    inp = torch.randn(1, 3, 128, 128) 
    out = model(inp)
    print("Output shape:", out.shape)
    summary(model, input_size=(1, 3, 128, 128))

    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name())