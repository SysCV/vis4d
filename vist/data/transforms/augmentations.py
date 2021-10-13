"""VisT augmentations."""
from typing import Sequence

from vist.struct import DictStrAny, Images
from vist.struct.labels import Bitmasks

from .base import BaseAugmentation


class VisTResize(BaseAugmentation):
    """VisT resize augmentation class."""

    def apply_mask(
        self, masks: Sequence[Bitmasks], parameters: DictStrAny
    ) -> Sequence[Bitmasks]:
        """Apply augmentation to input mask."""
        interp = self.augmentor.interpolation
        if interp != "nearest":
            self.augmentor.interpolation = "nearest"
        super().apply_mask(masks, parameters)
        self.augmentor.interpolation = interp
        return masks


class VisTColorJitter(BaseAugmentation):
    """VisT color jitter augmentation class."""

    def apply_image(self, image: Images, parameters: DictStrAny) -> Images:
        """Apply augmentation to input image."""
        imaget, tm = self.augmentor(
            image.tensor / 255.0, parameters, return_transform=True
        )
        parameters["transform_matrix"] = tm
        return Images(
            (imaget * 255).type(image.tensor.dtype),
            [(imaget.shape[3], imaget.shape[2])],
        )

    def apply_mask(
        self, masks: Sequence[Bitmasks], parameters: DictStrAny
    ) -> Sequence[Bitmasks]:
        """Skip augmentation for mask."""
        return masks
