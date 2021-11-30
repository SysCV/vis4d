"""Wrapper for conv2d."""
from typing import Optional

import torch
from torch import nn

from .conv2d import Conv2d


class BasicBlock(nn.Module):  # type: ignore
    """Basic build block."""

    def __init__(
        self,
        conv_in_dim: int,
        conv_out_dim: int,
        conv_has_bias: bool = False,
        is_downsample: bool = False,
        norm_cfg: Optional[str] = "BatchNorm2d",
    ):
        """Init."""
        super().__init__()
        self.is_downsample = is_downsample
        if norm_cfg is not None:
            norm = getattr(nn, norm_cfg)
        else:
            norm = None  # pragma: no cover
        if is_downsample:
            self.conv1 = Conv2d(
                conv_in_dim,
                conv_out_dim,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=conv_has_bias,
                norm=norm(conv_out_dim) if norm is not None else norm,
                activation=nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = Conv2d(
                conv_in_dim,
                conv_out_dim,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=conv_has_bias,
                norm=norm(conv_out_dim) if norm is not None else norm,
                activation=nn.ReLU(inplace=True),
            )
        self.conv2 = Conv2d(
            conv_out_dim,
            conv_out_dim,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=conv_has_bias,
            norm=norm(conv_out_dim) if norm is not None else norm,
        )

        if is_downsample:
            self.downsample = Conv2d(
                conv_in_dim,
                conv_out_dim,
                kernel_size=1,
                stride=2,
                bias=conv_has_bias,
                norm=norm(conv_out_dim) if norm is not None else norm,
            )
        elif conv_in_dim != conv_out_dim:
            self.downsample = Conv2d(  # pragma: no cover
                conv_in_dim,
                conv_out_dim,
                kernel_size=1,
                stride=1,
                bias=conv_has_bias,
                norm=norm(conv_out_dim) if norm is not None else norm,
            )
            self.is_downsample = True

        self.relu = nn.ReLU(True)

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        identity = input_x

        out = self.conv1(input_x)

        out = self.conv2(out)

        if self.is_downsample:
            identity = self.downsample(input_x)

        out += identity

        out = self.relu(out)

        return out
