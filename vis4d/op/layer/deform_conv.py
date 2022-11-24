"""Wrapper for deformable convolution."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.ops import DeformConv2d


class DeformConv(nn.Module):
    """Deformable Convolution operator.

    Includes batch normalization and ReLU activation.
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
        """Init.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (tuple[int, int], optional): Size of convolutional
                kernel. Defaults to (3, 3).
            stride (int, optional): Stride of convolutional layer. Defaults to
                1.
            padding (int, optional): Padding of convolutional layer. Defaults
                to 1.
            dilation (int, optional): Dilation of convolutional layer. Defaults
                to 1.
            groups (int, optional): Number of deformable groups. Defaults to 1.
            bias (bool, optional): Whether to use bias in convolutional layer.
                Defaults to True.
            bn_momentum (float, optional): Momentum of batch normalization.
                Defaults to 0.1.
        """
        super().__init__()
        self.conv_offset = nn.Conv2d(
            in_channels,
            groups * 3 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.actf = nn.Sequential(
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights of offset conv layer."""
        if hasattr(self, "conv_offset"):
            self.conv_offset.weight.data.zero_()
            if self.conv_offset.bias is not None:
                self.conv_offset.bias.data.zero_()

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.conv_offset(input_x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        input_x = self.deform_conv(input_x, offset, mask)
        input_x = self.actf(input_x)
        return input_x
