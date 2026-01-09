#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Context Aggregation Block (CAB)
-----------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


class DWConv(nn.Module):
    """Depthwise convolution block."""

    def __init__(self, channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------------------------------------------------------
# Pyramid Spatial Pooling
# -----------------------------------------------------------------------------


class PSPModule(nn.Module):
    """
    Pyramid Spatial Pooling with residual feature preservation.

    Input : (B, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(self, in_channels, sizes=(1, 3, 6, 8)):
        super().__init__()
        self.stages = nn.ModuleList([nn.AdaptiveAvgPool2d((s, s)) for s in sizes])

        self.project = nn.Conv2d(
            in_channels * (len(sizes) + 1),  # +1 for identity
            in_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        h, w = x.shape[2:]
        priors = [x]
        for pool in self.stages:
            pooled = pool(x)
            pooled = F.interpolate(
                pooled, size=(h, w), mode="bilinear", align_corners=False
            )
            priors.append(pooled)

        out = torch.cat(priors, dim=1)
        return self.project(out)


# -----------------------------------------------------------------------------
# Global Attention (Non-local + PSP)
# -----------------------------------------------------------------------------


class GlobalContextAttention(nn.Module):
    """
    Reduced Non-Local Attention with PSP-enhanced key/value encoding.
    """

    def __init__(
        self,
        in_channels,
        key_channels,
        value_channels,
        out_channels=None,
        scale=1,
        psp_sizes=(1, 3, 6, 8),
    ):
        super().__init__()

        self.scale = scale
        self.out_channels = out_channels or in_channels

        # Optional spatial reduction
        self.pool = nn.MaxPool2d(kernel_size=scale) if scale > 1 else nn.Identity()

        # Query / Key / Value projections
        self.to_query = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )

        self.to_key = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )

        self.to_value = nn.Conv2d(in_channels, value_channels, 1, bias=False)

        # Independent PSP encoders
        self.psp_key = PSPModule(key_channels, psp_sizes)
        self.psp_value = PSPModule(value_channels, psp_sizes)

        # Output projection (zero-init for stability)
        self.project_out = nn.Conv2d(
            value_channels, self.out_channels, kernel_size=1, bias=False
        )
        nn.init.constant_(self.project_out.weight, 0)

    def forward(self, x):
        B, _, H, W = x.shape
        x_down = self.pool(x)
        _, _, Hd, Wd = x_down.shape

        # Query: (B, HW, K)
        query = self.to_query(x_down)
        query = query.view(B, -1, Hd * Wd).transpose(1, 2)

        # Key: (B, K, Hd*Wd)
        key = self.psp_key(self.to_key(x_down))
        key = key.view(B, -1, Hd * Wd)

        # Value: (B, Hd*Wd, V)
        value = self.psp_value(self.to_value(x_down))
        value = value.view(B, -1, Hd * Wd).transpose(1, 2)

        # Attention
        attn = torch.bmm(query, key)
        attn = attn * (key.shape[1] ** -0.5)
        attn = F.softmax(attn, dim=-1)

        context = torch.bmm(attn, value)
        context = context.transpose(1, 2).view(B, -1, Hd, Wd)
        context = self.project_out(context)

        if self.scale > 1:
            context = F.interpolate(
                context, size=(H, W), mode="bilinear", align_corners=False
            )

        return context


# -----------------------------------------------------------------------------
# Local Attention
# -----------------------------------------------------------------------------


class LocalAttention(nn.Module):
    """Local spatial-channel refinement."""

    def __init__(self, channels):
        super().__init__()
        self.refine = nn.Sequential(
            DWConv(channels),
            DWConv(channels),
            DWConv(channels),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x):
        mask = self.gate(self.refine(x))
        return x + x * mask


# -----------------------------------------------------------------------------
# Context Aggregation Block (CAB)
# -----------------------------------------------------------------------------


class ContextAggregationBlock(nn.Module):
    """
    Final CAB module combining:
    - Global long-range reasoning
    - Local boundary refinement
    """

    def __init__(self, in_channels, value_channels):
        super().__init__()

        self.global_attn = GlobalContextAttention(
            in_channels=in_channels,
            key_channels=in_channels // 2,
            value_channels=value_channels,
            out_channels=in_channels,
            scale=1,
        )

        self.local_attn = LocalAttention(in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        global_feat = self.gamma * self.global_attn(x)
        local_feat = self.local_attn(x)
        return global_feat + local_feat


# -----------------------------------------------------------------------------
# Sanity check
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    model = ContextAggregationBlock(512, 128).cuda().eval()
    x = torch.randn(2, 512, 32, 64).cuda()
    with torch.no_grad():
        y = model(x)
    print("OK", x.shape, "→", y.shape)
    assert x.shape == y.shape
