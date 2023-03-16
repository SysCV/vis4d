"""Pad transformation."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vis4d.common.typing import NDArrayF32
from vis4d.data.const import CommonKeys as K

from .base import BatchTransform


@BatchTransform(K.images, K.images)
class PadImages:
    """Pad batch of images at the bottom right."""

    def __init__(
        self, stride: int = 32, mode: str = "constant", value: float = 0.0
    ) -> None:
        """Creates an instance of PadImage.

        Args:
            stride (int, optional): Chooses padding size so that the input will
                be divisible by stride. Defaults to 32.
            mode (str, optional): Padding mode. One of constant, reflect,
                replicate or circular. Defaults to "constant".
            value (float, optional): Value for constant padding.
                Defaults to 0.0.
        """
        self.stride = stride
        self.mode = mode
        self.value = value

    def __call__(self, images: list[NDArrayF32]) -> list[NDArrayF32]:
        """Pad images to consistent size."""
        heights = [im.shape[1] for im in images]
        widths = [im.shape[2] for im in images]
        max_hw = max(heights), max(widths)

        # ensure divisibility by stride
        def _make_divisible(x: int):
            return (x + (self.stride - 1)) // self.stride * self.stride

        max_hw = tuple(_make_divisible(x) for x in max_hw)

        # generate params for torch pad
        for i, (image, h, w) in enumerate(zip(images, heights, widths)):
            pad_param = (0, max_hw[1] - w, 0, max_hw[0] - h)
            image = torch.from_numpy(image).permute(0, 3, 1, 2)
            image = F.pad(image, pad_param, self.mode, self.value)
            images[i] = image.permute(0, 2, 3, 1).numpy()
        return images
