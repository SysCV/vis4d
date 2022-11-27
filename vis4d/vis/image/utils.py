"""Utility functions for image processing operations."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vis4d.common.typing import NDArrayBool, NDArrayNumber
from vis4d.vis.util import DEFAULT_COLOR_MAPPING

ImageType = npt.NDArray[np.float64]


def preprocess_masks(
    masks: NDArrayBool,
    class_ids: NDArrayNumber | None,
    color_mapping: list[tuple[float, float, float]] = DEFAULT_COLOR_MAPPING,
) -> tuple[list[NDArrayBool], list[tuple[float, float, float]]]:
    """Preprocesses predicted semantic masks.

    Args:
        masks (NDArrayBool): The semantic masks of shape [N, h, w].
        class_ids (NDArrayNumber, None):  An array with class ids for each mask
            shape [N]
        color_mapping (list[tuple[float, float, float]]): Color mapping for
            each semantic class

    Returns:
        tuple[list[masks], list[colors]]: Returns a list with all masks of
            shape [h,w] as well as a list with the corresponding colors.
    """
    mask_list: list[NDArrayBool] = []
    color_list: list[tuple[float, float, float]] = []

    for idx in range(masks.shape[0]):
        mask = masks[idx, ...]

        class_id = None if class_ids is None else class_ids[idx].item()
        if class_id is not None:
            color = color_mapping[class_id % len(color_mapping)]
        else:
            color = color_mapping[idx % len(color_mapping)]
        mask_list.append(mask)
        color_list.append(color)
    return mask_list, color_list


def preprocess_image(
    image: NDArrayNumber, mode: str = "RGB"
) -> npt.NDArray[np.uint8]:
    """Validate and convert input image.

    Args:
        image: CHW or HWC image (ImageType) with C = 3.
        mode: input channel format (e.g. BGR, HSV).

    Returns:
        np.array[uint8]: Processed image in RGB.
    """
    assert len(image.shape) == 3
    assert image.shape[0] == 3 or image.shape[-1] == 3

    # Convert torch to numpy convention
    if not image.shape[-1] == 3:
        image = image.transpose(1, 2, 0)

    # Convert image to [0, 255]
    min_val, max_val = (np.min(image, axis=(0, 1)), np.max(image, axis=(0, 1)))
    image = image.astype(np.float32)
    image = (image - min_val) / (max_val - min_val) * 255.0

    if mode == "BGR":
        image = image[..., [2, 1, 0]]
        mode = "RGB"

    return image.astype(np.uint8)
