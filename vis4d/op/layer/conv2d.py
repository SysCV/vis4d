"""Wrapper for conv2d."""
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class Conv2d(torch.nn.Conv2d):  # type: ignore
    """Wrapper around Conv2d to support empty inputs and norm/activation."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        """Init."""
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(  # pylint: disable=arguments-renamed
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        if not torch.jit.is_scripting():
            # https://github.com/pytorch/pytorch/issues/12013
            if (
                x.numel() == 0
                and self.training
                and isinstance(self.norm, torch.nn.SyncBatchNorm)
            ):
                raise ValueError(
                    "SyncBatchNorm does not support empty inputs!"
                )

        x = F.conv2d(
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
    norm_cfg: Optional[str],
    num_groups: int,
) -> Tuple[nn.ModuleList, int]:
    """Init conv branch for head."""
    convs = nn.ModuleList()
    if norm_cfg is not None:
        norm = getattr(nn, norm_cfg)
    else:
        norm = None
    if norm == nn.GroupNorm:
        norm = lambda x: nn.GroupNorm(num_groups, x)
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
