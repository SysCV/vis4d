"""Normalize Transform."""
from __future__ import annotations

import torch

from vis4d.common.typing import NDArrayF32
from vis4d.data.const import CommonKeys as K

from .base import BatchTransform, Transform


@Transform(K.images, K.images)
class NormalizeImage:
    """Normalize image tensor (range 0 to 255) with given mean and std."""

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

    def __call__(self, image: NDArrayF32) -> NDArrayF32:
        """Normalize image tensor."""
        img = torch.from_numpy(image).permute(0, 3, 1, 2)
        pixel_mean = torch.tensor(self.mean).view(-1, 1, 1)
        pixel_std = torch.tensor(self.std).view(-1, 1, 1)
        img = (img - pixel_mean) / (pixel_std + self.epsilon)
        return img.permute(0, 2, 3, 1).numpy()


@BatchTransform(K.images, K.images)
class BatchNormalizeImages:
    """Normalize batch of image tensor."""

    def __init__(
        self,
        mean: tuple[float, float, float] = (123.675, 116.28, 103.53),
        std: tuple[float, float, float] = (58.395, 57.12, 57.375),
        epsilon: float = 1e-08,
    ):
        """Creates an instance of BatchNormalizeImages.

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
        """Normalize images tensor."""
        transform = NormalizeImage(
            in_keys=["img"],
            out_keys=["img"],
            sensors=self.sensors,
            mean=self.mean,
            std=self.std,
            epsilon=self.epsilon,
        )
        for i, img in enumerate(images):
            images[i] = transform(img)
        return images
