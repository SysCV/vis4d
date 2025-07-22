"""Photometric transforms."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from torch import Tensor

from vis4d.common.imports import OPENCV_AVAILABLE
from vis4d.common.typing import NDArrayF32
from vis4d.data.const import CommonKeys as K

from .base import Transform

if OPENCV_AVAILABLE:
    import cv2
else:
    raise ImportError("cv2 is not installed.")


@Transform(K.images, K.images)
class RandomGamma:
    """Apply Gamma transformation to images.

    Args:
        gamma_range (tuple[float, float]): Range of gamma values.
        image_channel_mode (str, optional): Image channel mode. Defaults to
            "RGB".
    """

    def __init__(
        self,
        gamma_range: tuple[float, float] = (1.0, 1.0),
        image_channel_mode: str = "RGB",
    ) -> None:
        """Init function for Gamma."""
        self.gamma_range = gamma_range
        self.image_channel_mode = image_channel_mode
        assert image_channel_mode in {"RGB", "BGR"}, (
            "image_channel_mode should be 'RGB' or 'BGR', "
            f"got {image_channel_mode}."
        )

    def __call__(self, images: list[NDArrayF32]) -> list[NDArrayF32]:
        """Call function for Gamma transformation."""
        factor = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        return _adjust_images(
            images, TF.adjust_gamma, factor, self.image_channel_mode
        )


@Transform(K.images, K.images)
class RandomBrightness:
    """Apply Brightness transformation to images.

    Args:
        brightness_range (tuple[float, float]): Range of brightness values.
        image_channel_mode (str, optional): Image channel mode. Defaults to
            "RGB".
    """

    def __init__(
        self,
        brightness_range: tuple[float, float] = (1.0, 1.0),
        image_channel_mode: str = "RGB",
    ) -> None:
        """Init function for Brightness."""
        self.brightness_range = brightness_range
        self.image_channel_mode = image_channel_mode
        assert image_channel_mode in {"RGB", "BGR"}, (
            "image_channel_mode should be 'RGB' or 'BGR', "
            f"got {image_channel_mode}."
        )

    def __call__(self, images: list[NDArrayF32]) -> list[NDArrayF32]:
        """Call function for Brightness transformation."""
        factor = np.random.uniform(
            self.brightness_range[0], self.brightness_range[1]
        )
        return _adjust_images(
            images, TF.adjust_brightness, factor, self.image_channel_mode
        )


@Transform(K.images, K.images)
class RandomContrast:
    """Apply Contrast transformation to images.

    Args:
        contrast_range (tuple[float, float]): Range of contrast values.
        image_channel_mode (str, optional): Image channel mode. Defaults to
            "RGB".
    """

    def __init__(
        self,
        contrast_range: tuple[float, float] = (1.0, 1.0),
        image_channel_mode: str = "RGB",
    ):
        """Init function for Contrast."""
        self.contrast_range = contrast_range
        self.image_channel_mode = image_channel_mode
        assert image_channel_mode in {"RGB", "BGR"}, (
            "image_channel_mode should be 'RGB' or 'BGR', "
            f"got {image_channel_mode}."
        )

    def __call__(self, images: list[NDArrayF32]) -> list[NDArrayF32]:
        """Call function for Contrast transformation."""
        factor = np.random.uniform(
            self.contrast_range[0], self.contrast_range[1]
        )
        return _adjust_images(
            images, TF.adjust_contrast, factor, self.image_channel_mode
        )


@Transform(K.images, K.images)
class RandomSaturation:
    """Apply saturation transformation to images.

    Args:
        saturation_range (tuple[float, float]): Range of saturation values.
        image_channel_mode (str, optional): Image channel mode. Defaults to
            "RGB".
    """

    def __init__(
        self,
        saturation_range: tuple[float, float] = (1.0, 1.0),
        image_channel_mode: str = "RGB",
    ):
        """Init function for saturation."""
        self.saturation_range = saturation_range
        self.image_channel_mode = image_channel_mode
        assert image_channel_mode in {"RGB", "BGR"}, (
            "image_channel_mode should be 'RGB' or 'BGR', "
            f"got {image_channel_mode}."
        )

    def __call__(self, images: list[NDArrayF32]) -> list[NDArrayF32]:
        """Call function for saturation transformation."""
        factor = np.random.uniform(
            self.saturation_range[0], self.saturation_range[1]
        )
        return _adjust_images(
            images, TF.adjust_saturation, factor, self.image_channel_mode
        )


@Transform(K.images, K.images)
class RandomHue:
    """Apply hue transformation to images.

    Args:
        hue_range (tuple[float, float]): Range of hue values.
        image_channel_mode (str, optional): Image channel mode. Defaults to
            "RGB".
    """

    def __init__(
        self,
        hue_range: tuple[float, float] = (0.0, 0.0),
        image_channel_mode: str = "RGB",
    ):
        """Init function for hue."""
        self.hue_range = hue_range
        self.image_channel_mode = image_channel_mode
        assert image_channel_mode in {"RGB", "BGR"}, (
            "image_channel_mode should be 'RGB' or 'BGR', "
            f"got {image_channel_mode}."
        )

    def __call__(self, images: list[NDArrayF32]) -> list[NDArrayF32]:
        """Call function for Hue transformation."""
        factor = np.random.uniform(self.hue_range[0], self.hue_range[1])
        return _adjust_images(
            images, TF.adjust_hue, factor, self.image_channel_mode
        )


@Transform(K.images, K.images)
class ColorJitter:
    """Apply color jitter to images.

    Args:
        brightness_range (tuple[float, float]): Range of brightness values.
        contrast_range (tuple[float, float]): Range of contrast values.
        saturation_range (tuple[float, float]): Range of saturation values.
        hue_range (tuple[float, float]): Range of hue values.
        image_channel_mode (str, optional): Image channel mode. Defaults to
            "RGB".
    """

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.875, 1.125),
        contrast_range: tuple[float, float] = (0.5, 1.5),
        saturation_range: tuple[float, float] = (0.5, 1.5),
        hue_range: tuple[float, float] = (-0.05, 0.05),
        image_channel_mode: str = "RGB",
    ):
        """Init function for color jitter."""
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.image_channel_mode = image_channel_mode
        assert image_channel_mode in {"RGB", "BGR"}, (
            "image_channel_mode should be 'RGB' or 'BGR', "
            f"got {image_channel_mode}."
        )

    def __call__(self, images: list[NDArrayF32]) -> list[NDArrayF32]:
        """Call function for Hue transformation."""
        transform_order = np.random.permutation(4)
        for transform in transform_order:
            # apply photometric transforms in a random order
            if transform == 0:
                # random brightness
                bfactor = np.random.uniform(
                    self.brightness_range[0], self.brightness_range[1]
                )
                images = _adjust_images(
                    images,
                    TF.adjust_brightness,
                    bfactor,
                    self.image_channel_mode,
                )
            elif transform == 1:
                # random contrast
                cfactor = np.random.uniform(
                    self.contrast_range[0], self.contrast_range[1]
                )
                images = _adjust_images(
                    images,
                    TF.adjust_contrast,
                    cfactor,
                    self.image_channel_mode,
                )
            elif transform == 2:
                # random saturation
                sfactor = np.random.uniform(
                    self.saturation_range[0], self.saturation_range[1]
                )
                images = _adjust_images(
                    images,
                    TF.adjust_saturation,
                    sfactor,
                    self.image_channel_mode,
                )
            elif transform == 3:
                # random hue
                hfactor = np.random.uniform(
                    self.hue_range[0], self.hue_range[1]
                )
                images = _adjust_images(
                    images, TF.adjust_hue, hfactor, self.image_channel_mode
                )
        return images


def _adjust_images(
    images: list[NDArrayF32],
    adjust_func: Callable[[Tensor, float], Tensor],
    adj_factor: float,
    image_channel_mode: str = "RGB",
) -> list[NDArrayF32]:
    """Apply color transformation to images.

    Args:
        images (list[NDArrayF32]): Image to be transformed.
        adjust_func (Callable[[Tensor, float], Tensor]): Function to apply.
        adj_factor (float): Adjustment factor.
        image_channel_mode (str, optional): Image channel mode. Defaults to
            "RGB".

    Returns:
        list[NDArrayF32]: Transformed image.
    """
    for i, image in enumerate(images):
        if image_channel_mode == "BGR":
            image = image[..., [2, 1, 0]]  # convert to RGB
        image_ = torch.from_numpy(image).permute(0, 3, 1, 2) / 255.0
        image_ = adjust_func(image_, adj_factor) * 255.0
        images[i] = image_.permute(0, 2, 3, 1).numpy()
        if image_channel_mode == "BGR":
            images[i] = images[i][..., [2, 1, 0]]  # convert back to BGR
    return images


@Transform(K.images, K.images)
class RandomHSV:
    """Apply HSV transformation to images.

    Used by YOLOX. Modifed from: https://github.com/Megvii-BaseDetection/YOLOX.

    Args:
        hue_delta (int): Delta for hue.
        saturation_delta (int): Delta for saturation.
        value_delta (int): Delta for value.
        image_channel_mode (str, optional): Image channel mode. Defaults to
            "BGR".
    """

    def __init__(
        self,
        hue_delta: int = 5,
        saturation_delta: int = 30,
        value_delta: int = 30,
        image_channel_mode: str = "BGR",
    ):
        """Init function for HSV transformation."""
        assert OPENCV_AVAILABLE, "RandomHSV requires OpenCV to be installed."
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta
        self.image_channel_mode = image_channel_mode
        assert image_channel_mode in {"RGB", "BGR"}, (
            "image_channel_mode should be 'RGB' or 'BGR', "
            f"got {image_channel_mode}."
        )

    # pylint: disable=no-member
    def __call__(self, images: list[NDArrayF32]) -> list[NDArrayF32]:
        """Call function for Hue transformation."""
        for i, image in enumerate(images):
            image = image[0].astype(np.uint8)
            if self.image_channel_mode == "BGR":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image = image.astype(np.int16)
            hsv_gains = np.random.uniform(-1, 1, 3) * [
                self.hue_delta,
                self.saturation_delta,
                self.value_delta,
            ]
            # random selection of h, s, v
            hsv_gains = (hsv_gains * np.random.randint(0, 2, 3)).astype(
                np.int16
            )
            image[..., 0] = (image[..., 0] + hsv_gains[0]) % 180
            image[..., 1] = np.clip(image[..., 1] + hsv_gains[1], 0, 255)
            image[..., 2] = np.clip(image[..., 2] + hsv_gains[2], 0, 255)
            image = image.astype(np.uint8)
            if self.image_channel_mode == "BGR":
                cv2.cvtColor(image, cv2.COLOR_HSV2BGR, dst=image)
            else:
                cv2.cvtColor(image, cv2.COLOR_HSV2RGB, dst=image)
            images[i] = image[None, ...].astype(np.float32)
        return images
