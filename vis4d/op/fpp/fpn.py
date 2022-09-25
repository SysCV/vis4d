"""Feature Pyramid Network.

This is based on
`"Feature Pyramid Network for Object Detection"
<https://arxiv.org/abs/1612.03144>`_.
"""

from collections import OrderedDict
from typing import List

import torch
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
