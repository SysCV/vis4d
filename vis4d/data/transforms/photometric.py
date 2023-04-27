"""Photometric transforms."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import Tensor

from vis4d.common.typing import NDArrayF32
from vis4d.data.const import CommonKeys as K

from .base import Transform


@Transform(K.images, K.images)
class RandomGamma:
    """Apply Gamma transformation to image."""

    def __init__(self, gamma_range: tuple[float, float] = (1.0, 1.0)) -> None:
        """Init function for Gamma.

        Args:
            gamma_range (tuple[float, float]): Range of gamma values.
        """
        self.gamma_range = gamma_range

    def __call__(self, image: NDArrayF32) -> NDArrayF32:
        """Call function for Gamma transformation."""
        factor = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        return _adjust_image(image, TF.adjust_gamma, factor)


@Transform(K.images, K.images)
class RandomBrightness:
    """Apply Brightness transformation to image."""

    def __init__(
        self, brightness_range: tuple[float, float] = (1.0, 1.0)
    ) -> None:
        """Init function for Brightness.

        Args:
            brightness_range (tuple[float, float]): Range of brightness values.
        """
        self.brightness_range = brightness_range

    def __call__(self, image: NDArrayF32) -> NDArrayF32:
        """Call function for Brightness transformation."""
        factor = np.random.uniform(
            self.brightness_range[0], self.brightness_range[1]
        )
        return _adjust_image(image, TF.adjust_brightness, factor)


@Transform(K.images, K.images)
class RandomContrast:
    """Apply Contrast transformation to image.

    Args:
        contrast_range (tuple[float, float]): Range of contrast values.
    """

    def __init__(self, contrast_range: tuple[float, float] = (1.0, 1.0)):
        """Init function for Contrast."""
        self.contrast_range = contrast_range

    def __call__(self, image: NDArrayF32) -> NDArrayF32:
        """Call function for Contrast transformation."""
        factor = np.random.uniform(
            self.contrast_range[0], self.contrast_range[1]
        )
        return _adjust_image(image, TF.adjust_contrast, factor)


@Transform(K.images, K.images)
class RandomSaturation:
    """Apply saturation transformation to image.

    Args:
        saturation_range (tuple[float, float]): Range of saturation values.
    """

    def __init__(self, saturation_range: tuple[float, float] = (1.0, 1.0)):
        """Init function for saturation."""
        self.saturation_range = saturation_range

    def __call__(self, image: NDArrayF32) -> NDArrayF32:
        """Call function for saturation transformation."""
        factor = np.random.uniform(
            self.saturation_range[0], self.saturation_range[1]
        )
        return _adjust_image(image, TF.adjust_saturation, factor)


@Transform(K.images, K.images)
class RandomHue:
    """Apply hue transformation to image.

    Args:
        hue_range (tuple[float, float]): Range of hue values.
    """

    def __init__(self, hue_range: tuple[float, float] = (0.0, 0.0)):
        """Init function for hue."""
        self.hue_range = hue_range

    def __call__(self, image: NDArrayF32) -> NDArrayF32:
        """Call function for Hue transformation."""
        factor = np.random.uniform(self.hue_range[0], self.hue_range[1])
        return _adjust_image(image, TF.adjust_hue, factor)


@Transform(K.images, K.images)
class ColorJitter:
    """Apply color jitter to image."""

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.875, 1.125),
        contrast_range: tuple[float, float] = (0.5, 1.5),
        saturation_range: tuple[float, float] = (0.5, 1.5),
        hue_range: tuple[float, float] = (-0.05, 0.05),
    ):
        """Init function for hue."""
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range

    def __call__(self, image: NDArrayF32) -> NDArrayF32:
        """Call function for Hue transformation."""
        transform_order = np.random.permutation(4)
        for transform in transform_order:
            # apply photometric transforms in a random order
            if transform == 0:
                # random brightness
                factor = np.random.uniform(
                    self.brightness_range[0], self.brightness_range[1]
                )
                image = _adjust_image(image, TF.adjust_brightness, factor)
            elif transform == 1:
                # random contrast
                factor = np.random.uniform(
                    self.contrast_range[0], self.contrast_range[1]
                )
                image = _adjust_image(image, TF.adjust_contrast, factor)
            elif transform == 2:
                # random saturation
                factor = np.random.uniform(
                    self.saturation_range[0], self.saturation_range[1]
                )
                image = _adjust_image(image, TF.adjust_saturation, factor)
            elif transform == 3:
                # random hue
                factor = np.random.uniform(
                    self.hue_range[0], self.hue_range[1]
                )
                image = _adjust_image(image, TF.adjust_hue, factor)
        return image


def _adjust_image(
    image: NDArrayF32,
    adjust_func: Callable[[Tensor, float], Tensor],
    adj_factor: float,
) -> NDArrayF32:
    """Apply color transformation to image."""
    image_ = torch.from_numpy(image).permute(0, 3, 1, 2)
    image_ = adjust_func(image_, adj_factor).to(image_.dtype)
    return image_.permute(0, 2, 3, 1).numpy()
