"""Wrappers for torch vision backbones."""

from typing import List

import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from .base import Backbone


class ResNet(Backbone):
    """
    Wrapper for torch vision resnet backbones.
    """

    def __init__(
        self, name: str, pretrained: bool = True, trainable_layers: int = 5
    ):
        """Initiazlie the ResNet backbone from torch vision.

        We don't need the final pooled layer as the return of the backbone is
        supposed to be the feature pyramid.
        """
        super().__init__()
        self.backbone = resnet_fpn_backbone(
            name, pretrained=pretrained, trainable_layers=trainable_layers
        )

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Torchvision ResNet forward.

        Args:
            images (Tensor[N, C, H, W]): Image input to process. Expected to
                type float32 with values ranging 0..255.

        Returns:
            fp (List[torch.Tensor]): The output feature pyramid. The list index
            represents the level, which has a downsampling raio of 2^index.
            fp[0] and fp[1] is a reference to the input images and torchvision
            resnet downsamples the feature maps by 4 directly.
        """
        outs = [images, images, *self.backbone(images).values()]
        return outs  # remove the last pooled layer
