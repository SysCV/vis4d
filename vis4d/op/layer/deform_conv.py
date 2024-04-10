"""Wrapper for deformable convolution."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.ops import DeformConv2d

from .weight_init import constant_init


class DeformConv(DeformConv2d):  # type: ignore
    """Wrapper around Deformable Convolution operator with norm/activation.

    If norm is specified, it is initialized with 1.0 and bias with 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm: nn.Module | None = None,
        activation: nn.Module | None = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (int): Size of convolutional kernel.
            stride (int, optional): Stride of convolutional layer. Defaults to
                1.
            padding (int, optional): Padding of convolutional layer. Defaults
                to 0.
            dilation (int, optional): Dilation of convolutional layer. Defaults
                to 1.
            groups (int, optional): Number of deformable groups. Defaults to 1.
            bias (bool, optional): Whether to use bias in convolutional layer.
                Defaults to True.
            norm (nn.Module, optional): Normalization layer. Defaults to None.
            activation (nn.Module, optional): Activation layer. Defaults to
                None.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.norm = norm
        self.activation = activation
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights of offset conv layer."""
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()  # type: ignore
        if self.norm is not None:
            constant_init(self.norm, 1.0, bias=0.0)

    def forward(  # pylint: disable=arguments-differ
        self, input_x: Tensor
    ) -> Tensor:
        """Forward."""
        out = self.conv_offset(input_x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        input_x = super().forward(input_x, offset, mask)
        if self.norm is not None:
            input_x = self.norm(input_x)
        if self.activation is not None:
            input_x = self.activation(input_x)
        return input_x
