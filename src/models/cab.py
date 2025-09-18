#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class PSPModule(nn.Module):
    """Pyramid Spatial Pooling Module.

    Aggregates context at multiple scales.
    Input: (N, C, H, W)
    Output: (N, C, H, W) — fused multi-scale features
    """

    def __init__(self, sizes=(1, 3, 6, 8), in_channels=None):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList(
            [nn.AdaptiveAvgPool2d((size, size)) for size in sizes]
        )
        # Optional: Use 1x1 conv to reduce channel count after pooling
        self.conv_out = nn.Conv2d(len(sizes) * in_channels, in_channels, kernel_size=1)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = []
        for pool in self.stages:
            pooled = pool(feats)
            upsampled = F.interpolate(
                pooled, size=(h, w), mode="bilinear", align_corners=False
            )
            priors.append(upsampled)
        out = torch.cat(priors, dim=1)
        return self.conv_out(out)


class ReducedGlobalAttention(nn.Module):
    """Compact Global Attention Block with Multi-Scale Context Encoding.

    Applies non-local style attention using pyramid pooling.
    """

    def __init__(
        self,
        in_channels,
        key_channels,
        value_channels,
        out_channels=None,
        scale=1,
        psp_size=(1, 3, 6, 8),
    ):
        super(ReducedGlobalAttention, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        # Downsample input if needed
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale)) if scale > 1 else None

        # Key/Query transform (shared weights OK per paper)
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )

        # Value transform
        self.f_value = nn.Conv2d(in_channels, value_channels, kernel_size=1, bias=False)

        # Final projection
        self.W = nn.Conv2d(value_channels, self.out_channels, kernel_size=1, bias=False)

        # PSP for global context on keys/values
        self.psp = PSPModule(sizes=psp_size, in_channels=value_channels)

        # Initialize projection to 0 to preserve identity initially
        nn.init.constant_(self.W.weight, 0)
        if self.W.bias is not None:
            nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, c, h, w = x.shape

        # Optional downsampling
        if self.pool is not None:
            x_down = self.pool(x)
        else:
            x_down = x

        n_d, c_d, h_d, w_d = x_down.shape

        # Query: (B, K, H*W)
        query = self.f_query(x_down).view(n_d, self.key_channels, -1)
        query = query.permute(0, 2, 1).contiguous()  # (B, H*W, K)

        # Key: (B, K, S₁*S₁ + S₂*S₂ + ...) via PSP
        key = self.f_key(x_down)
        key_psp = self.psp(key)  # (B, K, H_d, W_d)
        key = key_psp.view(n_d, self.key_channels, -1)  # (B, K, H_d*W_d)

        # Value: (B, V, H_d, W_d) → (B, V, H_d*W_d)
        value = self.f_value(x_down)
        value = self.psp(value)  # Apply PSP to enhance context
        value = value.view(n_d, self.value_channels, -1)  # (B, V, H_d*W_d)
        value = value.permute(0, 2, 1).contiguous()  # (B, H_d*W_d, V)

        # Attention map: (B, H*W, H_d*W_d)
        sim_map = torch.bmm(query, key)  # (B, H*W, H_d*W_d)
        sim_map = (self.key_channels**-0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)  # Attend over downsampled space

        # Output: (B, H*W, V)
        context = torch.bmm(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()  # (B, V, H*W)
        context = context.view(n_d, self.value_channels, h, w)  # (B, V, H, W)

        # Project back to input dimension
        context = self.W(context)

        # Upsample if downsampled
        if self.scale > 1:
            context = F.interpolate(
                context, size=(h, w), mode="bilinear", align_corners=False
            )

        return context


class LocalAttention(nn.Module):
    """Local Channel-Spatial Refinement Block.

    Enhances features using depthwise convs and sigmoid gating.
    """

    def __init__(self, inplane):
        super(LocalAttention, self).__init__()
        self.dwconv = nn.Sequential(
            _DWConv(inplane, inplane, stride=1),  # Keep spatial size
            _DWConv(inplane, inplane, stride=1),
            _DWConv(inplane, inplane, stride=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        mask = self.dwconv(x)
        mask = self.sigmoid(mask)
        out = x * mask
        return out + residual  # Add modulated features back


class ContextAggregationBlock(nn.Module):
    """Combines Global and Local Attention.

    Global branch captures long-range dependencies. Local branch refines boundaries and
    fine details.
    """

    def __init__(self, inplane, plane):
        super(ContextAggregationBlock, self).__init__()
        self.global_attn = ReducedGlobalAttention(
            in_channels=inplane,
            key_channels=inplane // 2,
            value_channels=plane,
            out_channels=inplane,
            scale=1,
            psp_size=(1, 3, 6, 8),
        )
        self.local_attn = LocalAttention(inplane)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable weight for global path

    def forward(self, x):
        # Global context path
        global_feat = self.global_attn(x)
        global_feat = global_feat * self.gamma

        # Local refinement path
        local_feat = self.local_attn(x)

        # Combine
        return global_feat + local_feat


# Reuse consistent _DWConv (same as in cabinet.py)
class _DWConv(nn.Module):
    """Depthwise Convolution."""

    def __init__(self, dw_channels, out_channels, stride=1):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    # Test full CAB
    model = ContextAggregationBlock(512, 128)  # inplane=512, plane=128
    model.eval()
    model.cuda()

    with torch.no_grad():
        inp = torch.randn(2, 512, 32, 64).cuda()
        out = model(inp)

    print(f"Input: {inp.shape} → Output: {out.shape}")
    assert out.shape == inp.shape, "Output shape must match input"
    print("ContextAggregationBlock test passed!")
