#!/usr/bin/python
# -*- encoding: utf-8 -*-

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cab import ContextAggregationBlock
from src.models.mobilenetv3 import mobilenetv3_small as MobileNetV3


class _DWConv(nn.Module):
    """Depthwise Convolution."""

    def __init__(self, dw_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolution."""

    def __init__(self, dw_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False
            ),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)  # Save memory
        self.init_weight()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv.weight, a=1)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)


class AttentionBranch(nn.Module):
    def __init__(
        self, inplanes: int, interplanes: int, outplanes: int, num_classes: int
    ):
        super().__init__()
        self.conva = nn.Sequential(
            nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
            nn.BatchNorm2d(interplanes),
            nn.ReLU(True),
        )
        self.a2block = ContextAggregationBlock(interplanes, interplanes // 2)
        self.convb = nn.Conv2d(interplanes, outplanes, kernel_size=1, bias=True)

        # Fusion path
        self.b1 = nn.Conv2d(inplanes + outplanes, outplanes, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(outplanes)
        self.b3 = nn.ReLU(True)
        self.b4 = nn.Conv2d(outplanes, num_classes, kernel_size=1, bias=True)

        self.init_weight()

    def forward(self, x):
        feat = self.conva(x)
        feat = self.a2block(feat)
        low_res_out = self.convb(feat)  # This is feat_ab_final analog

        fused = torch.cat([x, feat], dim=1)
        fused = self.b1(fused)
        fused = self.b2(fused)
        fused = self.b3(fused)
        high_res_out = self.b4(fused)  # Final segmentation head

        return low_res_out, high_res_out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SpatialBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_out(x)
        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan: int, out_chan: int):
        super().__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()  # Correct: channel-wise attention

    def forward(self, fsp: torch.Tensor, fcp: torch.Tensor) -> torch.Tensor:
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)

        atten = self.avg_pool(feat)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)

        feat_atten = feat * atten
        return feat_atten + feat


class CABiNetOutput(nn.Module):
    def __init__(self, in_chan: int, mid_chan: int, n_classes: int):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return self.conv_out(x)


class CABiNet(nn.Module):
    def __init__(self, n_classes: int, backbone_weights: Optional[Path] = None):
        super().__init__()
        # Load MobileNetV3 backbone
        self.mobile = MobileNetV3(pretrained=False, width_mult=1.0)

        # Only load custom weights if provided
        if backbone_weights is not None:
            try:
                state_dict = torch.load(backbone_weights, map_location="cpu")
                # Filter only backbone keys (e.g., 'features.*')
                filtered_state_dict = {
                    k: v for k, v in state_dict.items() if k.startswith("features")
                }
                self.mobile.load_state_dict(filtered_state_dict, strict=False)
                print(f"[INFO] Loaded backbone weights from {backbone_weights}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load backbone weights from {backbone_weights}: {e}"
                )

        self.ab = AttentionBranch(576, 256, 256, n_classes)
        self.sb = SpatialBranch()
        self.ffm = FeatureFusionModule(
            128 + 256, 256
        )  # 128 from sb, 256 from ab-upsampled
        self.conv_out = CABiNetOutput(256, 256, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W = x.shape[2:]
        # device = x.device

        # Extract features
        feat_sb = self.sb(x)  # (B, 128, H/8, W/8)
        mobile_feat = self.mobile(x)  # (B, 576, H/16, W/16)

        # Attention branch
        low_res_logit, high_res_logit = self.ab(
            mobile_feat
        )  # (B, 256, H/16, W/16), (B, C, H/16, W/16)

        # Upsample attention outputs to spatial branch resolution
        low_res_logit_up = F.interpolate(
            low_res_logit, size=feat_sb.shape[2:], mode="bilinear", align_corners=False
        )
        high_res_logit_up = F.interpolate(
            high_res_logit, size=feat_sb.shape[2:], mode="bilinear", align_corners=False
        )

        # Fuse features
        feat_fuse = self.ffm(feat_sb, low_res_logit_up)  # Now both at H/8, W/8
        final_logit = self.conv_out(feat_fuse)

        # Upsample to original resolution
        final_logit = F.interpolate(
            final_logit, (H, W), mode="bilinear", align_corners=False
        )
        high_res_logit_up = F.interpolate(
            high_res_logit_up, (H, W), mode="bilinear", align_corners=False
        )

        return final_logit, high_res_logit_up

    def get_params(self):
        """
        Returns separate parameter groups for:
        - WD: regular weights
        - No-WD: biases, BN params
        - LR-multiplied WD/No-WD (for decoder)
        """
        wd_params = []
        nowd_params = []

        lr_mul_wd = []
        lr_mul_nowd = []

        for name, child in self.named_children():
            if name in ("ffm", "conv_out", "ab"):
                # Decoder parts: use higher LR
                for m in child.modules():
                    if isinstance(m, nn.Conv2d):
                        lr_mul_wd.append(m.weight)
                        if m.bias is not None:
                            lr_mul_nowd.append(m.bias)
                    elif isinstance(m, nn.BatchNorm2d):
                        lr_mul_nowd.extend(list(m.parameters()))
            else:
                # Backbone & spatial branch
                for m in child.modules():
                    if isinstance(m, nn.Conv2d):
                        wd_params.append(m.weight)
                        if m.bias is not None:
                            nowd_params.append(m.bias)
                    elif isinstance(m, nn.BatchNorm2d):
                        nowd_params.extend(list(m.parameters()))

        return wd_params, nowd_params


if __name__ == "__main__":
    backbone_path = Path(
        "src/models/pretrained_backbones/mobilenetv3-small-55df8e1f.pth"
    )

    if not backbone_path.exists():
        print(
            f"Warning: Backbone weights not found at {backbone_path},"
            " training from scratch."
        )
        backbone_path = None

    net = CABiNet(n_classes=19, backbone_weights=backbone_path)
    net.eval()

    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out, out16 = net(x)

    print("Output shapes:", out.shape, out16.shape)
    assert out.shape[-2:] == (512, 512), "Final output must match input size"
    print("✅ CABiNet forward pass successful!")
