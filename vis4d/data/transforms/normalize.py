"""Normalize Transform."""
from __future__ import annotations

import torch

from .base import Transform


@Transform()
def normalize_image(
    mean: tuple[float, float, float] = (123.675, 116.28, 103.53),
    std: tuple[float, float, float] = (58.395, 57.12, 57.375),
    epsilon: float = 1e-08,
):
    """Normalize image tensor (assumed range 0..255) with given mean and std.

    Args:
        mean (Tuple[float, float, float], optional): Mean value. Defaults to
            (123.675, 116.28, 103.53).
        std (Tuple[float, float, float], optional): Standard deviation value.
            Defaults to (58.395, 57.12, 57.375).
        epsilon (float, optional): Epsilon for numerical stability of division.
            Defaults to 1e-08.
    """

    def _normalize(img: torch.Tensor) -> torch.Tensor:
        pixel_mean = torch.tensor(mean, device=img.device).view(-1, 1, 1)
        pixel_std = torch.tensor(std, device=img.device).view(-1, 1, 1)
        img = (img - pixel_mean) / (pixel_std + epsilon)
        return img

    return _normalize
