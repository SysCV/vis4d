"""Normalize Transform."""

from __future__ import annotations

import torch

from vis4d.common.typing import NDArrayF32

from ..const import CommonKeys as K
from .base import Transform


@Transform(K.images, K.images)
class NormalizeImages:
    """Normalize a list of image tensor with given mean and std.

    Image tensor is of shape [N, H, W, C] and range (0, 255).
    """

    def __init__(
        self,
        mean: tuple[float, float, float] = (123.675, 116.28, 103.53),
        std: tuple[float, float, float] = (58.395, 57.12, 57.375),
        epsilon: float = 1e-08,
    ) -> None:
        """Creates an instance of NormalizeImage.

        Args:
            mean (Tuple[float, float, float], optional): Mean value. Defaults
                to (123.675, 116.28, 103.53).
            std (Tuple[float, float, float], optional): Standard deviation
                value. Defaults to (58.395, 57.12, 57.375).
            epsilon (float, optional): Epsilon for numerical stability of
                division. Defaults to 1e-08.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def __call__(self, images: list[NDArrayF32]) -> list[NDArrayF32]:
        """Normalize image tensor."""
        for i, image in enumerate(images):
            img = torch.from_numpy(image).permute(0, 3, 1, 2)
            pixel_mean = torch.tensor(self.mean).view(-1, 1, 1)
            pixel_std = torch.tensor(self.std).view(-1, 1, 1)
            img = (img - pixel_mean) / (pixel_std + self.epsilon)

            images[i] = img.permute(0, 2, 3, 1).numpy()

        return images
