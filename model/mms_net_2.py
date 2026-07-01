import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F


from .path_block_a import PathBlockA
from .path_block_b import PathBlockB
from .path_block_c import PathBlockC
from .path_block_d import PathBlockD
from .path_block_e import PathBlockE
from .path_block_f import PathBlockF
from .se import ChannelSpatialSELayer
from .reverse_attention import ReverseAttention
from .multi_head_attention import MultiHeadAttention
# from .cbam import CBAM
from .bgm import BoundaryGuidedModule

# -----------------------------
# 🔹 MMSNet - 6 branches
# 🔹 RA
# 🔹 SE
# 🔹 Multi-Head Attention
# 🔹 Boundary Learning Module
# -----------------------------

# -----------------------------
# 🔹 Basic Blocks
# -----------------------------
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


class DeconvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


def make_ra_head(in_ch, mid_ch, out_ch):
    return nn.Sequential(
        ConvBNReLU(in_ch, mid_ch),
        ConvBNReLU(mid_ch, mid_ch),
        nn.Conv2d(mid_ch, out_ch, kernel_size=1)
    )

def make_bgm_head(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1)
    )

import torch.nn as nn

class MHA(nn.Module):
    def __init__(self, in_channels, n_head):
        super().__init__()

        self.mha = MultiHeadAttention(
            n_head=n_head,
            d_model=in_channels,
            d_k=in_channels // n_head,
            d_v=in_channels // n_head
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Flatten
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        # Attention
        out, _ = self.mha(x_flat, x_flat, x_flat)
        # Restore
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)

        return out

# -----------------------------
# 🔹 Main Model
# -----------------------------
class MMSNet(nn.Module):
    def __init__(self, in_ch=16, bn_size=1, out_ch=1):
        super().__init__()

        # Channel config 
        c1 = in_ch
        c2 = in_ch * 2
        c4 = in_ch * 4
        c12 = in_ch * 12

        # -----------------
        # Input stem
        # -----------------
        self.stem = ConvBNReLU(3, c1)

        # -----------------
        # Path blocks
        # -----------------
        self.paths_stage1 = nn.ModuleList([
            PathBlockA(in_ch=c1),
            PathBlockB(in_ch=c1),
            PathBlockC(in_ch=c1),
        ])

        self.paths_stage2 = nn.ModuleList([
            PathBlockD(in_ch=c1),
            PathBlockE(in_ch=c1),
            PathBlockF(in_ch=c1),
        ])

        # -----------------
        # SE blocks
        # -----------------
        self.se_stage1 = nn.ModuleList([
            ChannelSpatialSELayer(c4),
            ChannelSpatialSELayer(c4),
            ChannelSpatialSELayer(c4),
        ])

        self.se_stage2 = nn.ModuleList([
            ChannelSpatialSELayer(c2),
            ChannelSpatialSELayer(c2),
            ChannelSpatialSELayer(c2),
        ])

        # self.cbam = CBAM(channels= c12 + c1, r=8)

        # -----------------
        # Bottleneck
        # -----------------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c12 + c1, c1 * bn_size, 1, bias=False),
            nn.BatchNorm2d(c1 * bn_size),
            nn.ReLU(inplace=True),
        )

        self.attention = MHA(in_channels=c1 * bn_size, n_head=4)

        self.up1 = DeconvBNReLU(c1 * bn_size, c2)
        self.up2 = DeconvBNReLU(c2, c1)

        # -----------------
        # Mid stage
        # -----------------
        self.mid_up = DeconvBNReLU(c2 + c1 * 6, c2)

        # -----------------
        # Reverse Attention
        # -----------------
        self.ra1 = ReverseAttention(c12 + c1)
        self.ra2 = ReverseAttention(c1 * 6 + c2)

        self.ra1_head = make_ra_head(c12 + c1, c2, out_ch)
        self.ra2_head = make_ra_head(c1 * 6 + c2, c2, out_ch)

        # -----------------
        # Boundary Guided Module
        # -----------------

        self.bgm1_head = make_bgm_head(c12 + c1, c2)
        self.bgm2_head = make_bgm_head(c1 * 6 + c2, c2)
        # pre conv1x1 before CB module
            
        self.bgm_conv = nn.Conv2d(c2 * 2, c2 * 2, kernel_size=1)
        self.bgm = BoundaryGuidedModule(c2 * 2)
        
        # -----------------
        # Output
        # -----------------
        self.out_head = nn.Conv2d(c2, out_ch, 1)

    # -----------------------------
    def resize(self, x, scale):
        return F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)

    # -----------------------------
    def forward(self, x):

        # -------- Stem --------
        x = self.stem(x)
        skip1 = x

        # -------- Stage 1 --------
        paths1 = [se(p(x)) for p, se in zip(self.paths_stage1, self.se_stage1)]

        skip1_ds = self.resize(skip1, 0.25)
        fused1 = torch.cat(paths1 + [skip1_ds], dim=1)

        # fused1 = self.cbam(fused1)
        # -------- Bottleneck --------
        x = self.bottleneck(fused1)
        # x = self.attention(x)
        z = self.up1(x)
        x = self.up2(z)

        # -------- Stage 2 --------
        paths2 = [se(p(x)) for p, se in zip(self.paths_stage2, self.se_stage2)]
        fused2 = torch.cat(paths2 + [z], dim=1)

        x = self.mid_up(fused2)

        # -------- Output 3 --------
        out3 = self.out_head(x)

        # -------- Output 2 --------
        # RA 2
        out3_ds = self.resize(out3, 0.5)
        ra2_feat = self.ra2(fused2, out3_ds)
        logit2 = self.ra2_head(ra2_feat) + out3_ds
        out2 = self.resize(logit2, 2)

        # -------- Output 1 --------
        # RA 1
        out2_ds = self.resize(out2, 0.25)
        ra1_feat = self.ra1(fused1, out2_ds)
        logit1 = self.ra1_head(ra1_feat) + out2_ds
        out1 = self.resize(logit1, 4)

        # BGM
        bgm_out2 = self.resize(fused2, 2)
        bgm_feat2 = self.bgm2_head(bgm_out2)

        bgm_out1 = self.resize(fused1, 4)
        bgm_feat1 = self.bgm1_head(bgm_out1)

        # CB
        bgm_fused = torch.cat([bgm_feat1, bgm_feat2], dim=1)
        bgm_fused = self.bgm_conv(bgm_fused) 
        out_bgm = self.bgm(bgm_fused)

        return out1, out2, out3, out_bgm
    
if __name__ == "__main__":
    model = MMSNet()
    inp = torch.randn(1, 3, 128, 128) 
    out1, out2, out3, out_bgm = model(inp)

    print("Output 1 shape:", out1.shape)
    print("Output 2 shape:", out2.shape)
    print("Output 3 shape:", out3.shape)
    print("Output bgm shape:", out_bgm.shape)

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
