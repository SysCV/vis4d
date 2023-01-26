"""Pad transformation."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BatchTransform


@BatchTransform()
def pad_image(stride: int = 32, mode: str = "constant", value: float = 0.0):
    """Pad batch of images at the bottom right.

    Args:
        stride (int, optional): Chooses padding size so that the input will be
            divisible by stride. Defaults to 32.
        mode (str, optional): Padding mode. One of constant, reflect,
            replicate or circular. Defaults to "constant".
        value (float, optional): Value for constant padding. Defaults to 0.0.
    """

    def _pad(images: list[torch.Tensor]) -> list[torch.Tensor]:
        heights = [im.shape[-2] for im in images]
        widths = [im.shape[-1] for im in images]
        max_hw = max(heights), max(widths)

        # ensure divisibility by stride
        def _make_divisible(x: int):
            return (x + (stride - 1)) // stride * stride

        max_hw = tuple(_make_divisible(x) for x in max_hw)

        # generate params for torch pad
        for i, (image, h, w) in enumerate(zip(images, heights, widths)):
            pad_param = (0, max_hw[1] - w, 0, max_hw[0] - h)
            images[i] = F.pad(image, pad_param, mode, value)
        return images

    return _pad
