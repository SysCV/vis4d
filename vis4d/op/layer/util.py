"""Utility functions for layer ops."""
from __future__ import annotations

from torch import nn

from .deform_conv import DeformConv


def build_conv_layer(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = False,
    use_dcn: bool = False,
) -> nn.Module:
    """Build a convolution layer."""
    if use_dcn:
        return DeformConv(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )
