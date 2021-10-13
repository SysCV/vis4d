"""VisT augmentations."""
from typing import Sequence

import torch
from vist.struct import DictStrAny, Images
from vist.struct.labels import Bitmasks

from .base import KorniaAugmentationWrapper


class KorniaColorJitter(KorniaAugmentationWrapper):
    """Wrapper for Kornia color jitter augmentation class."""

    def apply_image(
        self, image: Images, parameters: DictStrAny, transform: torch.Tensor
    ) -> Images:
        """Apply augmentation to input image."""
        imaget = self.apply_transform(
            image.tensor / 255.0, parameters, transform
        )
        return Images(
            (imaget * 255).type(image.tensor.dtype),
            [(imaget.shape[3], imaget.shape[2])],
        )

    def apply_mask(
        self,
        masks: Sequence[Bitmasks],
        parameters: DictStrAny,
        transform: torch.Tensor,
    ) -> Sequence[Bitmasks]:
        """Skip augmentation for mask."""
        return masks
