"""Common reusable neural network layers for CABiNet models."""

import torch
import torch.nn as nn


class DepthwiseConv(nn.Module):
    """Depthwise Convolution layer.

    Applies a depthwise convolution followed by batch normalization and ReLU activation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for convolution. Default: 1
        kernel_size: Kernel size for convolution. Default: 3
        padding: Padding for convolution. Default: 1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Output tensor of shape (N, C', H', W')
        """
        return self.conv(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution.

    Consists of a depthwise convolution followed by a pointwise (1x1) convolution.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for depthwise convolution. Default: 1
        kernel_size: Kernel size for depthwise convolution. Default: 3
        padding: Padding for depthwise convolution. Default: 1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Output tensor of shape (N, C', H', W')
        """
        return self.conv(x)
