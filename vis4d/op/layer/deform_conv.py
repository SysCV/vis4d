"""Wrapper for deformable convolution."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.ops import DeformConv2d


class DeformConv(nn.Module):
    """Deformable Convolution operator.

    Includes batchnorm and ReLU activation and sane defaults.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        bn_momentum: float = 0.1,
    ) -> None:
        """Init."""
        super().__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        input_x = self.conv(input_x)
        input_x = self.actf(input_x)
        return input_x
