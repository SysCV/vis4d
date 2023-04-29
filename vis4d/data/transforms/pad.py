"""Pad transformation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from vis4d.common.typing import NDArrayF32, NDArrayUI8
from vis4d.data.const import CommonKeys as K

from .base import BatchTransform


@BatchTransform(K.images, [K.images, K.input_hw])
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
    ) -> tuple[list[NDArrayF32], list[tuple[int, int]]]:
        """Pad images to consistent size."""
        heights = [im.shape[1] for im in images]
        widths = [im.shape[2] for im in images]
        if self.shape is not None:
            max_hw = self.shape
            out_shapes = [max_hw] * len(images)
        else:
            max_hw = max(heights), max(widths)
            max_hw = tuple(_make_divisible(x, self.stride) for x in max_hw)  # type: ignore # pylint: disable=line-too-long
            out_shapes = [(im.shape[1], im.shape[2]) for im in images]

        # generate params for torch pad
        for i, (image, h, w) in enumerate(zip(images, heights, widths)):
            pad_param = (0, max_hw[1] - w, 0, max_hw[0] - h)
            image_ = torch.from_numpy(image).permute(0, 3, 1, 2)
            image_ = F.pad(image_, pad_param, self.mode, self.value)
            images[i] = image_.permute(0, 2, 3, 1).numpy()
        return images, out_shapes


@BatchTransform(K.seg_masks, K.seg_masks)
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

    def __call__(self, masks: list[NDArrayUI8]) -> list[NDArrayUI8]:
        """Pad images to consistent size."""
        heights = [im.shape[0] for im in masks]
        widths = [im.shape[1] for im in masks]
        if self.shape is not None:
            max_hw = self.shape
        else:
            max_hw = max(heights), max(widths)
            max_hw = tuple(_make_divisible(x, self.stride) for x in max_hw)  # type: ignore # pylint: disable=line-too-long

        # generate params for torch pad
        for i, (mask, h, w) in enumerate(zip(masks, heights, widths)):
            pad_param = ((0, max_hw[0] - h), (0, max_hw[1] - w))
            masks[i] = np.pad(
                mask, pad_param, mode=self.mode, constant_values=self.value
            )
        return masks


def _make_divisible(x: int, stride: int) -> int:
    """Ensure divisibility by stride."""
    return (x + (stride - 1)) // stride * stride
