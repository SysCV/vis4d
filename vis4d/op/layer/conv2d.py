"""Wrapper for conv2d."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vis4d.common.typing import ArgsType

from .weight_init import constant_init


class Conv2d(nn.Conv2d):
    """Wrapper around Conv2d to support empty inputs and norm/activation."""

    def __init__(
        self,
        *args: ArgsType,
        norm: nn.Module | None = None,
        activation: nn.Module | None = None,
        **kwargs: ArgsType,
    ) -> None:
        """Creates an instance of the class.

        If norm is specified, it is initialized with 1.0 and bias with 0.0.
        """
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

        if self.norm is not None:
            constant_init(self.norm, 1.0, bias=0.0)

    def forward(  # pylint: disable=arguments-renamed
        self, x: Tensor
    ) -> Tensor:
        """Forward pass."""
        if not torch.jit.is_scripting():  # type: ignore
            # https://github.com/pytorch/pytorch/issues/12013
            if (
                x.numel() == 0
                and self.training
                and isinstance(self.norm, nn.SyncBatchNorm)
            ):
                raise ValueError(
                    "SyncBatchNorm does not support empty inputs!"
                )

        x = F.conv2d(  # pylint: disable=not-callable
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def add_conv_branch(
    num_branch_convs: int,
    last_layer_dim: int,
    conv_out_dim: int,
    conv_has_bias: bool,
    norm_cfg: str | None,
    num_groups: int | None,
) -> tuple[nn.ModuleList, int]:
    """Init conv branch for head."""
    convs = nn.ModuleList()
    if norm_cfg is not None:
        norm = getattr(nn, norm_cfg)
    else:
        norm = None

    if norm == nn.GroupNorm:
        assert num_groups is not None, "num_groups must be specified"
        norm = lambda x: nn.GroupNorm(  # pylint: disable=unnecessary-lambda-assignment
            num_groups, x
        )
    if num_branch_convs > 0:
        for i in range(num_branch_convs):
            conv_in_dim = last_layer_dim if i == 0 else conv_out_dim
            convs.append(
                Conv2d(
                    conv_in_dim,
                    conv_out_dim,
                    kernel_size=3,
                    padding=1,
                    bias=conv_has_bias,
                    norm=norm(conv_out_dim) if norm is not None else norm,
                    activation=nn.ReLU(inplace=True),
                )
            )
        last_layer_dim = conv_out_dim

    return convs, last_layer_dim


class UnetDownConvOut(NamedTuple):
    """Output of the UnetDownConv operator.

    features: Features before applying the pooling operator
    pooled_features: Features after applying the pooling operator
    """

    features: Tensor
    pooled_features: Tensor


class UnetDownConv(nn.Module):
    """Downsamples a feature map by applying two convolutions and maxpool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling: bool = True,
        activation: str = "ReLU",
    ):
        """Creates a new downsampling convolution operator.

        This operator consists of two convolutions followed by a maxpool
        operator.

        Args:
            in_channels (int): input channesl
            out_channels (int): output channesl
            pooling (bool): If pooling should be applied
            activation (str): Activation that should be applied
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        activation = getattr(nn, activation)()

        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
        )

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def __call__(self, data: Tensor) -> UnetDownConvOut:
        """Applies the operator.

        Args:
            data (Tensor): Input data.

        Returns:
            UnetDownConvOut: Containing the features before the pooling
                operation (features) and after (pooled_features).
        """
        return self._call_impl(data)

    def forward(self, data: Tensor) -> UnetDownConvOut:
        """Applies the operator.

        Args:
            data (Tensor): Input data.

        Returns:
            UnetDownConvOut: containing the features before the pooling
                operation (features) and after (pooled_features).
        """
        x = F.relu(self.conv1(data))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return UnetDownConvOut(features=before_pool, pooled_features=x)


class UnetUpConv(nn.Module):
    """An operator that performs 2 convolutions and 1 UpConvolution.

    A ReLU activation follows each convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        merge_mode: str = "concat",
        up_mode: str = "transpose",
    ):
        """Creates a new UpConv operator.

        This operator merges two inputs by upsampling one and combining it with
        the other.

        Args:
            in_channels: Number of input channels (low res)
            out_channels: Number of output channels (high res)
            merge_mode: How to merge both input channels
            up_mode: How to upsample the channel with lower resolution

        Raises:
            ValueError: If upsampling mode is unknown
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        # Upsampling
        if self.up_mode == "transpose":
            self.upconv: nn.Module = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        elif self.up_mode == "upsample":
            self.upconv = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown upsampling mode: {up_mode}")

        if self.merge_mode == "concat":
            self.conv1 = nn.Conv2d(
                2 * self.out_channels, self.out_channels, 3, padding=1
            )
        else:
            # num of input channels to conv2 is same
            self.conv1 = nn.Conv2d(
                self.out_channels, self.out_channels, 3, padding=1
            )
        self.conv2 = nn.Conv2d(
            self.out_channels, self.out_channels, 3, padding=1
        )

    def __call__(self, from_down: Tensor, from_up: Tensor) -> Tensor:
        """Forward pass.

        Arguments:
            from_down (Tensor): Tensor from the encoder pathway. Assumed to
                have dimension 'out_channels'
            from_up (Tensor): Upconv'd tensor from the decoder pathway. Assumed
                to have dimension 'in_channels'
        """
        return self._call_impl(from_down, from_up)

    def forward(self, from_down: Tensor, from_up: Tensor) -> Tensor:
        """Forward pass.

        Arguments:
            from_down (Tensor): Tensor from the encoder pathway. Assumed to
                have dimension 'out_channels'
            from_up (Tensor): Upconv'd tensor from the decoder pathway. Assumed
                to have dimension 'in_channels'
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == "concat":
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x
