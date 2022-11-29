"""Utility functions for image processing operations."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vis4d.common.typing import NDArrayBool, NDArrayNumber
from vis4d.vis.image.base import CanvasBackend
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


def get_intersection_point(
    point1: tuple[float, float, float],
    point2: tuple[float, float, float],
    camera_near_clip: float,
) -> tuple[float, float]:
    """Get point intersecting with camera near plane on line point1 -> point2.

    The line is defined by two points in pixel coordinates and their depth.
    """
    c1, c2, c3 = 0, 0, camera_near_clip
    a1, a2, a3 = 0, 0, 1
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    k_up = abs(a1 * (x1 - c1) + a2 * (y1 - c2) + a3 * (z1 - c3))
    k_down = abs(a1 * (x1 - x2) + a2 * (y1 - y2) + a3 * (z1 - z2))
    if k_up > k_down:
        k = 1
    else:
        k = k_up / k_down
    return (1 - k) * point1 + k * point2


def draw_box3d(
    canvas: CanvasBackend,
    corners: tuple[tuple[float, float], ...],
    label: str,
    color: tuple[float, float, float],
) -> None:
    """Draw 3D bounding box on a given 2D canvas.

    Args:
        canvas (CanvasBackend): Current canvas to draw on.
        corners (tuple[tuple[float, float], ...]): Projected locations of the
            3D bounding box corners.
        label (str): Text label of the 3D box.
        color (tuple[float, float, float]): The box color.
    """
    assert len(corners) == 8, "A 3D box needs 8 corners."
    # Draw the sides
    for i in range(4):
        canvas.draw_line(corners[i], corners[i + 4], color)

    # Draw bottom (first 4 corners) and top (last 4 corners)
    canvas.draw_rotated_box(corners[:4], color)
    canvas.draw_rotated_box(corners[4:], color)

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners[:2], axis=0)
    center_bottom = np.mean(corners[:4], axis=0)
    canvas.draw_line(center_bottom, center_bottom_forward, color)

    # Draw label
    canvas.draw_text(corners[0], label, color)
