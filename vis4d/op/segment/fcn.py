"""Feature pyramid for semantic segmentation.

This is based on the implementation of
`Fully Convolution Networks for Semantic Segmentation
<https://arxiv.org/abs/1411.4038>`_.
"""

from typing import List

import torch
import torch.nn.functional as F

from .base import FeaturePyramidProcessing


class FCN(FeaturePyramidProcessing):
    """FPP for segmentation head."""

    def __init__(self, align_corners: bool = True) -> None:
        """Init.

        Args:
            align_corners (bool): If to align corners in upsampling. Default:
                True
        """
        super().__init__()
        self.align_corners = align_corners

    def _resize_and_concat_feats(
        self, feats: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Resize and concat the features.

        Args:
            feats (List[torch.Tensor]): List of multi-level img features.
        Returns:
            upsampled_feats (torch.Tensor): List of upsampled features.
        """
        upsampled_feats = [
            F.interpolate(
                input=feat,
                size=feats[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            for feat in feats
        ]
        return upsampled_feats

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward function for transforming feature maps and obtain
        segmentation prediction.

        Args:
            x (List[torch.Tensor]): List of multi-level image features.
        Returns:
            output (torch.Tensor): A tensor of shape (batch_size,
                self.channels, H, W) which is feature map for last layer of
                decoder head.
        """
        return self._resize_and_concat_feats(x)

    def __call__(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Type definition for call implementation."""
        return self._call_impl(x)
