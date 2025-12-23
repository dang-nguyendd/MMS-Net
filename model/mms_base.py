import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F

from .path_block_a import PathBlockA
from .path_block_b import PathBlockB
from .path_block_c import PathBlockC
from .path_block_d import PathBlockD
from .path_block_e import PathBlockE
from .feature_booster import FeatureBooster
from .se import ChannelSpatialSELayer


class MMSNet(nn.Module):
    """
    Full Path
    
    """
    def __init__(self, fb_ch = 8, in_ch = 16, bn_size = 2, out_ch =2):
        super().__init__()

        self.se_a = ChannelSpatialSELayer(num_channels=in_ch*4)
        self.se_b = ChannelSpatialSELayer(num_channels=in_ch*4)
        self.se_c = ChannelSpatialSELayer(num_channels=in_ch*4)
        self.se_d = ChannelSpatialSELayer(num_channels=in_ch*2)
        self.se_e = ChannelSpatialSELayer(num_channels=in_ch*2)

        self.path_a = PathBlockA(in_ch=in_ch)
        self.path_b = PathBlockB(in_ch=in_ch)
        self.path_c = PathBlockC(in_ch=in_ch)
        self.path_d = PathBlockD(in_ch=in_ch)
        self.path_e = PathBlockE(in_ch=in_ch)

        self.feature_booster_1 = FeatureBooster(in_ch=3, out_ch= fb_ch)
        self.feature_booster_2 = FeatureBooster(in_ch=in_ch*2, out_ch = fb_ch)

        self.bn = nn.BatchNorm2d(in_ch)
        self.bn1 = nn.BatchNorm2d(in_ch*bn_size)
        self.bn2 = nn.BatchNorm2d(in_ch*2)
        self.bn3 = nn.BatchNorm2d(in_ch)

        self.relu = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

        # Input stem
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=in_ch,
            kernel_size=3,
            padding=1
        )

        self.conv_1x1 = nn.Conv2d(
            in_channels=in_ch*12 + in_ch,
            out_channels=in_ch*bn_size,
            kernel_size=1,
            padding=0
        )

        self.de_conv_1 = nn.ConvTranspose2d(
            in_channels=in_ch*bn_size,
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
            in_channels=in_ch*4 + fb_ch + in_ch*2,
            out_channels=in_ch*2,
            kernel_size=2,
            stride=2
        )

        self.conv_1x1_2 = nn.Conv2d(
            in_channels=in_ch*2 + fb_ch,
            out_channels=out_ch,
            kernel_size=1,
            padding=0
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

        # self.conv_down_sample = nn.Sequential(
        #     nn.AvgPool2d(2, 2),
        #     nn.Conv2d(in_ch, in_ch, 3, padding=1),
        #     nn.BatchNorm2d(in_ch),        
        #     nn.ReLU(inplace=True), 
        #     nn.AvgPool2d(2, 2),
        #     nn.Conv2d(in_ch, in_ch, 3, padding=1),
        #     nn.BatchNorm2d(in_ch),        
        #     nn.ReLU(inplace=True), 
        # )

    def down_sample(self, in_map, out_map):
        x_resized = F.interpolate(
            in_map,
            size=out_map.size()[2:],
            mode='bilinear',
            align_corners=False
        )
        return x_resized

    def forward(self, x):
        fb_1 = self.feature_booster_1.forward(x)
        
        # Input stem
        x = self.conv(x)
        dense_skip_path_1 = x
        x = self.bn(x)
        x = self.relu(x)

        #__________ Cascaded Path 1 __________
        # Parallel branches
        path_a = self.path_a.forward(x)
        path_b = self.path_b.forward(x)
        path_c = self.path_c.forward(x)

        # Enhance with SE block
        path_a = self.se_a.forward(path_a)
        path_b = self.se_b.forward(path_b)
        path_c = self.se_c.forward(path_c)


        dense_skip_path_1 = self.down_sample(dense_skip_path_1, path_a)
        # dense_skip_path_1 = self.conv_down_sample(dense_skip_path_1)
                                               
        # Depth-wise (channel dimension) concatenation
        fused_1 = torch.cat([path_a, path_b, path_c, dense_skip_path_1], dim=1)

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

        # Enhance with SE block
        path_d = self.se_d.forward(path_d)
        path_e = self.se_e.forward(path_e)

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


    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())

    print("Trainable params:", trainable)
    print("Non-trainable params:", non_trainable)
    print("Non-trainable buffers:", buffers)
    print("Total params including buffers:", trainable + non_trainable + buffers)
