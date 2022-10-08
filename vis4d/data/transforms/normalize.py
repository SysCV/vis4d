"""Normalize Transform."""
from typing import Tuple

import torch

from .base import Transform


@Transform()
def normalize_image(
    mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
    std: Tuple[float, float, float] = (58.395, 57.12, 57.375),
):
    """Normalize image tensor with given mean and std."""

    def _normalize(img: torch.Tensor) -> torch.Tensor:
        pixel_mean = torch.tensor(mean, device=img.device).view(-1, 1, 1)
        pixel_std = torch.tensor(std, device=img.device).view(-1, 1, 1)
        img = (img - pixel_mean) / pixel_std
        return img

    return _normalize
