"""Pad transformation."""
from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F

from vis4d.common.typing import NDArrayF32, NDArrayUI8
from vis4d.data.const import CommonKeys as K

from .base import BatchTransform


class PadParam(TypedDict):
    """Parameters for Resize."""

    target_shape: tuple[int, int]


@BatchTransform(K.images, [K.images, "transforms.pad"])
class PadImages:
    """Pad batch of images at the bottom right."""

    def __init__(
        self,
        stride: int = 32,
        mode: str = "constant",
        value: float = 0.0,
        shape: tuple[int, int] | None = None,
    ) -> None:
        """Creates an instance of PadImage.

        Args:
            stride (int, optional): Chooses padding size so that the input will
                be divisible by stride. Defaults to 32.
            mode (str, optional): Padding mode. One of constant, reflect,
                replicate or circular. Defaults to "constant".
            value (float, optional): Value for constant padding.
                Defaults to 0.0.
            shape (tuple[int, int], optional): Shape of the padded image
                (H, W). Defaults to None.
        """
        self.stride = stride
        self.mode = mode
        self.value = value
        self.shape = shape

    def __call__(
        self, images: list[NDArrayF32]
    ) -> tuple[list[NDArrayF32], list[PadParam]]:
        """Pad images to consistent size."""
        heights = [im.shape[1] for im in images]
        widths = [im.shape[2] for im in images]
        max_hw = _get_max_shape(self.stride, self.shape, heights, widths)

        # generate params for torch pad
        for i, (image, h, w) in enumerate(zip(images, heights, widths)):
            pad_param = (0, max_hw[1] - w, 0, max_hw[0] - h)
            image_ = torch.from_numpy(image).permute(0, 3, 1, 2)
            image_ = F.pad(image_, pad_param, self.mode, self.value)
            images[i] = image_.permute(0, 2, 3, 1).numpy()

        pad_params = [PadParam(target_shape=max_hw)] * len(images)

        return images, pad_params


@BatchTransform([K.seg_masks, "transforms.pad.target_shape"], K.seg_masks)
class PadSegMasks:
    """Pad batch of segmentation masks at the bottom right."""

    def __init__(
        self,
        stride: int = 32,
        mode: str = "constant",
        value: int = 255,
        shape: tuple[int, int] | None = None,
    ) -> None:
        """Creates an instance of PadSegMasks.

        Args:
            stride (int, optional): Chooses padding size so that the input will
                be divisible by stride. Defaults to 32.
            mode (str, optional): Padding mode. One of constant, reflect,
                replicate or circular. Defaults to "constant".
            value (float, optional): Value for constant padding.
                Defaults to 0.0.
            shape (tuple[int, int], optional): Shape of the padded image
                (H, W). Defaults to None.
        """
        self.stride = stride
        self.mode = mode
        self.value = value
        self.shape = shape

    def __call__(
        self,
        masks: list[NDArrayUI8],
        target_shapes: list[tuple[int, int]] | None = None,
    ) -> list[NDArrayUI8]:
        """Pad images to consistent size."""
        heights = [im.shape[0] for im in masks]
        widths = [im.shape[1] for im in masks]
        if target_shapes is not None:
            max_hw = target_shapes[0]
        else:
            max_hw = _get_max_shape(self.stride, self.shape, heights, widths)

        # generate params for torch pad
        for i, (mask, h, w) in enumerate(zip(masks, heights, widths)):
            pad_param = ((0, max_hw[0] - h), (0, max_hw[1] - w))
            masks[i] = np.pad(  # type: ignore
                mask, pad_param, mode=self.mode, constant_values=self.value
            )
        return masks


def _get_max_shape(
    stride: int,
    shape: tuple[int, int] | None,
    heights: list[int],
    widths: list[int],
) -> tuple[int, int]:
    """Get max shape for padding.

    Args:
        stride (int): Chooses padding size so that the input will be divisible
            by stride.
        shape (tuple[int, int], optional): Shape of the padded image (H, W).
            Defaults to None.
        heights (list[int]): List of heights of input.
        widths (list[int]): List of widths of input.

    Returns:
        tuple[int, int]: Max shape for padding.
    """
    if shape is not None:
        max_hw = shape
    else:
        max_hw = max(heights), max(widths)
        max_hw = tuple(_make_divisible(x, stride) for x in max_hw)  # type: ignore # pylint: disable=line-too-long
    return max_hw


def _make_divisible(x: int, stride: int) -> int:
    """Ensure divisibility by stride."""
    return (x + (stride - 1)) // stride * stride
