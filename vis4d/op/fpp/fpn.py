"""Feature Pyramid Network.

This is based on `"Feature Pyramid Network for Object Detection"
<https://arxiv.org/abs/1612.03144>`_.
"""

from __future__ import annotations

from collections import OrderedDict

import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import FeaturePyramidNetwork as _FPN
from torchvision.ops.feature_pyramid_network import (
    ExtraFPNBlock as _ExtraFPNBlock,
)
from torchvision.ops.feature_pyramid_network import (
    LastLevelMaxPool,
)

from .base import FeaturePyramidProcessing


class FPN(_FPN, FeaturePyramidProcessing):  # type: ignore
    """Feature Pyramid Network.

    This is a wrapper of the torchvision implementation.
    """

    def __init__(
        self,
        in_channels_list: list[int],
        out_channels: int,
        extra_blocks: _ExtraFPNBlock | None = LastLevelMaxPool(),
        start_index: int = 2,
    ) -> None:
        """Init without additional components.

        Args:
            in_channels_list (list[int]): List of input channels.
            out_channels (int): Output channels.
            extra_blocks (_ExtraFPNBlock, optional): Extra block. Defaults to
                LastLevelMaxPool().
            start_index (int, optional): Start index of base model feature
                maps. Defaults to 2.
        """
        super().__init__(
            in_channels_list, out_channels, extra_blocks=extra_blocks
        )
        self.start_index = start_index

    def forward(self, x: list[Tensor]) -> list[Tensor]:
        """Process the input features with FPN.

        Because by default, FPN doesn't upsample the first two feature maps in
        the pyramid, we keep the first two feature maps intact.

        Args:
            x (list[Tensor]): Feature pyramid as outputs of the
            base model.

        Returns:
            list[Tensor]: Feature pyramid after FPN processing.
        """
        feat_dict = OrderedDict(
            (k, v)
            for k, v in zip(
                [str(i) for i in range(len(x) - self.start_index)],
                x[self.start_index :],
            )
        )
        outs = super().forward(feat_dict)  # type: ignore
        return [*x[: self.start_index], *outs.values()]  # type: ignore

    def __call__(self, x: list[Tensor]) -> list[Tensor]:
        """Type definition for call implementation."""
        return self._call_impl(x)


class ExtraFPNBlock(_ExtraFPNBlock):  # type: ignore
    """Extra block in the FPN.

    This is a wrapper of the torchvision implementation.
    """

    def __init__(
        self,
        extra_levels: int,
        in_channels: int,
        out_channels: int,
        add_extra_convs: str = "on_output",
        extra_relu: bool = False,
    ) -> None:
        """Create an instance of the class."""
        super().__init__()
        self.extra_levels = extra_levels
        self.add_extra_convs = add_extra_convs
        self.extra_relu = extra_relu

        self.convs = nn.ModuleList()
        if extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == "on_input":
                    _in_channels = in_channels
                else:
                    _in_channels = out_channels

                extra_fpn_conv = nn.Conv2d(
                    _in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                )
                self.convs.append(extra_fpn_conv)

    def forward(
        self, results: list[Tensor], x: list[Tensor], names: list[str]
    ) -> tuple[list[Tensor], list[str]]:
        """Forward."""
        if self.add_extra_convs == "on_input":
            extra_source = x[-1]
        elif self.add_extra_convs == "on_output":
            extra_source = results[-1]
        else:
            raise NotImplementedError

        results.append(self.convs[0](extra_source))
        names.append(str(int(names[-1]) + 1))

        for i in range(1, self.extra_levels):
            if self.extra_relu:
                results.append(self.convs[i](F.relu(results[-1])))
            else:
                results.append(self.convs[i](results[-1]))
            names.append(str(int(names[-1]) + 1))

        return results, names
