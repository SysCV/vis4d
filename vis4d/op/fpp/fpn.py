"""Feature Pyramid Network.

This is based on
`"Feature Pyramid Network for Object Detection"
<https://arxiv.org/abs/1612.03144>`_.
"""

from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import FeaturePyramidNetwork as _FPN
from torchvision.ops.feature_pyramid_network import (
    ExtraFPNBlock,
    LastLevelMaxPool,
)

from .base import FeaturePyramidProcessing


class FPN(_FPN, FeaturePyramidProcessing):
    """Feature Pyramid Network.

    This is a wrapper of the torchvision implementation.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: ExtraFPNBlock = LastLevelMaxPool(),
        start_index: int = 2,
    ):
        """Init without additional components."""
        super().__init__(
            in_channels_list, out_channels, extra_blocks=extra_blocks
        )
        self.start_index = start_index

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Process the input features with FPN.

        Because by default, FPN doesn't upsample the first two feature maps in
        the pyramid, we keep the first two feature maps intact.

        TODO(tobiasfshr) Add tests and use it in faster rcnn operation demo

        Args:
            x (List[torch.Tensor]): Feature pyramid as outputs of the
            base model.

        Returns:
            List[torch.Tensor]: Feature pyramid after FPN processing.
        """
        feat_dict = OrderedDict(
            (k, v)
            for k, v in zip(
                [str(i) for i in range(len(x) - self.start_index)],
                x[self.start_index :],
            )
        )
        outs = super().forward(feat_dict)
        return [*x[: self.start_index], *outs.values()]

    def __call__(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Type definition for call implementation."""
        return self._call_impl(x)


class LastLevelP6P7(ExtraFPNBlock):
    """This module is used in RetinaNet to generate extra layers, P6 and P7.

    Implementation modified from torchvision:
    https://github.com/pytorch/vision.
    Modified to add option for whether to use ReLU between additional layers.
    """

    def __init__(
        self, in_channels: int, out_channels: int, extra_relu: bool = False
    ):
        """Init."""
        super().__init__()
        self.extra_relu = extra_relu
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(
        self, p: List[torch.Tensor], c: List[torch.Tensor], names: List[str]
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Forward."""
        p5, c5 = p[-1], c[-1]
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        if self.extra_relu:
            p6 = F.relu(p6)
        p7 = self.p7(p6)
        p.extend([p6, p7])
        names.extend(["p6", "p7"])
        return p, names
