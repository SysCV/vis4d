"""Crop transformation."""
from __future__ import annotations

import torch
import torchvision.transforms.functional as V

from vis4d.common.typing import NDArrayF32
from vis4d.data.const import CommonKeys as K

from .base import Transform


@Transform(in_keys=(K.images,), out_keys=(K.images,))
class CenterCrop:
    """Crop the given image from the center."""

    def __init__(self, shape: tuple[int, int]):
        """Init transform.

        Args:
            shape (tuple[int, int]): Desired output shape.
        """
        self.shape = shape

    def __call__(self, image: NDArrayF32) -> NDArrayF32:
        """Randomly crop and resize the given image.

        Args:
            image (NDArrayF32): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        image_ = torch.from_numpy(image).permute(0, 3, 1, 2)
        crops = V.center_crop(image_, self.shape)
        return crops.permute(0, 2, 3, 1).numpy()
