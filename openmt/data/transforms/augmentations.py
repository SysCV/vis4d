"""openMT augmentations."""
import random
from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from detectron2.data.transforms import transform as T
from PIL import Image

from .base import BaseAugmentation, BaseKorniaAugmentation


class Resize(BaseKorniaAugmentation):
    """Simple reszie augmentation class."""

    def __init__(
        self,
        shape: Tuple[int, int],
        interp=None,
        return_transform: bool = False,
    ) -> None:
        """Init."""
        super().__init__(return_transform)
        self.new_h, self.new_w = shape
        if interp is None:
            self.interp = Image.BILINEAR
        self.return_transform = return_transform

    def __repr__(self) -> str:
        """Generate class name."""
        return self.__class__.__name__ + f"({super().__repr__()})"

    def compute_transformation(self, inputs, params):
        """Dummy one here."""
        return None

    def apply_transform(  # pylint: disable=arguments-renamed
        self, inputs, params, transform
    ):
        """Dummy one here."""
        return inputs

    def forward(  # pylint: disable=arguments-renamed
        self,
        img,
        params=None,
        return_transform=None,
    ) -> torch.Tensor:
        """Overwrite to use simple scaling and generate transform matrix."""
        interp_method = self.interp

        if return_transform is None:
            return_transform = self.return_transform

        h, w = img.shape[2:]

        pil_resize_to_interpolate_mode = {
            Image.NEAREST: "nearest",
            Image.BILINEAR: "bilinear",
            Image.BICUBIC: "bicubic",
        }
        mode = pil_resize_to_interpolate_mode[interp_method]
        align_corners = None if mode == "nearest" else False
        img = F.interpolate(
            img,
            (self.new_h, self.new_w),
            mode=mode,
            align_corners=align_corners,
        )

        # compute transformation
        transform = torch.eye(3, 3)

        transform[0][0] = self.new_w / w
        transform[1][1] = self.new_h / h

        if self.return_transform:
            return img, transform.unsqueeze(0)

        return img


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
