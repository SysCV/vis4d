"""Utility functions for layer ops."""

from __future__ import annotations

from torch import nn

from .conv2d import Conv2d
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
    norm: nn.Module | None = None,
    activation: nn.Module | None = None,
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
            norm=norm,
            activation=activation,
        )

    return Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        norm=norm,
        activation=activation,
    )


def build_activation_layer(
    activation: str, inplace: bool = False
) -> nn.Module:
    """Build activation layer.

    Args:
        activation (str): Activation layer type.
        inplace (bool, optional): If to set inplace. Defaults to False. It will
            be ignored if the activation layer is not inplace.
    """
    activation_layer = getattr(nn, activation)

    if activation_layer in {nn.Tanh, nn.PReLU, nn.Sigmoid, nn.GELU}:
        return activation_layer()

    return activation_layer(inplace=inplace)


def build_norm_layer(
    norm: str, out_channels: int, num_groups: int | None = None
) -> nn.Module:
    """Build normalization layer.

    Args:
        norm (str): Normalization layer type.
        out_channels (int): Number of output channels.
        num_groups (int | None, optional): Number of groups for GroupNorm.
            Defaults to None.
    """
    norm_layer = getattr(nn, norm)
    if norm_layer == nn.GroupNorm:
        assert (
            num_groups is not None
        ), "num_groups must be specified when using Group Norm"
        return norm_layer(num_groups, out_channels)

    return norm_layer(out_channels)
