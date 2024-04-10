"""Cross Stage Partial Layer.

Modified from mmdetection (https://github.com/open-mmlab/mmdetection).
"""

from __future__ import annotations

import torch
from torch import nn

from .conv2d import Conv2d


class DarknetBottleneck(nn.Module):
    """The basic bottleneck block used in Darknet.

    Each ResBlock consists of two Conv blocks and the input is added to the
    final output. Each block is composed of Conv, BN, and SiLU.
    The first convolutional layer has filter size of 1x1 and the second one
    has filter size of 3x3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (float, optional): The kernel size of the convolution.
            Defaults to 0.5.
        add_identity (bool, optional): Whether to add identity to the output.
            Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
    ):
        """Init."""
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv2d(
            in_channels,
            hidden_channels,
            1,
            bias=False,
            norm=nn.BatchNorm2d(hidden_channels, eps=0.001, momentum=0.03),
            activation=nn.SiLU(inplace=True),
        )
        self.conv2 = Conv2d(
            hidden_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            norm=nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03),
            activation=nn.SiLU(inplace=True),
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features (torch.Tensor): Input features.
        """
        identity = features
        out = self.conv1(features)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        return out


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float, optional): Ratio to adjust the number of channels
            of the hidden layer. Defaults to 0.5.
        num_blocks (int, optional): Number of blocks. Defaults to 1.
        add_identity (bool, optional): Whether to add identity in blocks.
            Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
    ):
        """Init."""
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = Conv2d(
            in_channels,
            mid_channels,
            1,
            bias=False,
            norm=nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.03),
            activation=nn.SiLU(inplace=True),
        )
        self.short_conv = Conv2d(
            in_channels,
            mid_channels,
            1,
            bias=False,
            norm=nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.03),
            activation=nn.SiLU(inplace=True),
        )
        self.final_conv = Conv2d(
            2 * mid_channels,
            out_channels,
            1,
            bias=False,
            norm=nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03),
            activation=nn.SiLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            *[
                DarknetBottleneck(
                    mid_channels, mid_channels, 1.0, add_identity
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features (torch.Tensor): Input features.
        """
        x_short = self.short_conv(features)

        x_main = self.main_conv(features)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)
        return self.final_conv(x_final)
