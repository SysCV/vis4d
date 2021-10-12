"""VisT augmentations."""
from typing import Sequence

from vist.struct import DictStrAny
from vist.struct.labels import Bitmasks

from .base import BaseAugmentation


class VisTResize(BaseAugmentation):
    """Simple resize augmentation class."""

    def apply_mask(
        self, masks: Sequence[Bitmasks], parameters: DictStrAny
    ) -> Sequence[Bitmasks]:
        """Apply augmentation to input mask."""
        if self.augmentor.interpolation != "nearest":
            self.augmentor.interpolation = "nearest"
        super().apply_mask(masks, parameters)
        return masks
