"""Utility functions for image processing operations."""
from __future__ import annotations

import colorsys

import numpy as np
import numpy.typing as npt

from vis4d.common.typing import NDArrayBool, NDArrayNumber

ImageType = npt.NDArray[np.float64]


def generate_color_map(length: int) -> list[tuple[float, float, float]]:
    """Generate a color palette of [length] colors."""
    brightness = 0.7
    hsv = [(i / length, 1, brightness) for i in range(length)]
    colors = [colorsys.hsv_to_rgb(*c) for c in hsv]
    colors = (np.array(colors) * 255).astype(np.uint8).tolist()
    s = np.random.get_state()
    np.random.seed(0)
    result = [tuple(colors[i]) for i in np.random.permutation(len(colors))]
    np.random.set_state(s)
    return result


DEFAULT_COLOR_MAPPING = generate_color_map(50)


def preprocess_masks(
    masks: NDArrayBool,
    class_ids: NDArrayNumber | None,
    color_mapping: list[tuple[float, float, float]] = DEFAULT_COLOR_MAPPING,
):
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
