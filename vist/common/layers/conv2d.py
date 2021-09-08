"""Wrapper for conv2d."""
import torch
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
