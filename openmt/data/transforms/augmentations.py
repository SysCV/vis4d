"""openMT augmentations."""
import random
from typing import Dict, Tuple, Union

from detectron2.data.transforms import transform as T

from .base import BaseAugmentation

# TODO switch to using kornia.augmentation here


class BrightnessJitterAugmentation(BaseAugmentation):
    """Simple brightness augmentation class.

    Multiplies color values by random factor alpha.
    """

    def __init__(self, brightness: float = 0.1):
        """Init."""
        super().__init__()
        self.brightness = brightness

    def get_transform(
        self, *args: Dict[str, Union[bool, float, str, Tuple[int, int]]]
    ) -> T.Transform:
        """Get deterministic transformation."""
        alpha = 1.0 + random.uniform(-self.brightness, self.brightness)
        return T.ColorTransform(lambda x: x * alpha)
